import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask, render_template, session, request, jsonify
from flask_socketio import SocketIO, emit
from joblib import load
import pandas as pd
from urllib.parse import urlparse
import re
import os
import user_manager
from datetime import datetime
import gdown

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt cảnh báo symlink bằng cách set biến môi trường:

# Khởi tạo Flask và SocketIO
app = Flask(__name__)
# print(os.urandom(24))
app.secret_key = os.environ.get('SECRET_KEY', 'default_fallback_key')
socketio = SocketIO(app)


class URLTransformerExtractor(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super(URLTransformerExtractor, self).__init__()
        # Tải mô hình BERT và tokenizer để xử lý văn bản
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Thêm một lớp phân loại đơn giản để phân loại URL
        self.url_detector = nn.Sequential(
            nn.Linear(768, 512),  # Dự đoán đầu ra từ BERT (768 chiều) và giảm xuống 512
            nn.ReLU(),  # Hàm kích hoạt ReLU
            nn.Dropout(0.2),  # Dropout để tránh overfitting
            nn.Linear(512, 256),  # Dự đoán tiếp với 256 chiều
            nn.ReLU(),  # Hàm kích hoạt ReLU
            nn.Dropout(0.2),  # Dropout tiếp
            nn.Linear(256, 2)  # Kết quả cuối cùng phân loại thành 2 lớp (phishing hay không)
        )

        self.max_length = 512  # Độ dài tối đa của chuỗi đầu vào cho BERT

    def forward(self, text):
        # Token hóa văn bản và truyền vào mô hình BERT
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_attention_mask=True)
        # Thực hiện dự đoán với BERT
        outputs = self.bert(**inputs)
        sequence_output = outputs.last_hidden_state
        # Phân loại kết quả thông qua lớp phân loại
        url_logits = self.url_detector(sequence_output)
        return url_logits


class PhishingDetector:
    def __init__(self):
        # Google Drive ID của file
        self.drive_file_id = "1-40zEVlUvpwPoP8vFMYLGWmgwEpDl_xm"
        self.local_model_path = "phishing_model.pkl"

        # Tải model nếu chưa có
        self.download_model_from_drive()

        # Load pre-trained model
        self.model = load(self.local_model_path)

    def download_model_from_drive(self):
        if not os.path.exists(self.local_model_path):
            print("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={self.drive_file_id}"
            gdown.download(url, self.local_model_path, quiet=False)
            print("Download complete!")

    # Hàm rút trích đặc trưng từ URL
    def extract_features(self, url):
        features = []

        if isinstance(url, str):
            # Phân tích URL
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname if parsed_url.hostname else ''

            # 1. Độ dài của URL
            features.append(len(url))

            # 2. Độ dài của hostname
            features.append(len(hostname))

            # 3. Địa chỉ IP trong URL
            features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)

            # 4-18. Tổng số ký tự đặc biệt trong URL (từ . đến $)
            special_chars = ['.', '-', '@', '?', '&', '=', '_', '~', '%',
                             '/', '*', ':', ',', ';', '$']
            for char in special_chars:
                features.append(len(re.findall(re.escape(char), url)))

            # 19. Số lần xuất hiện 'www'
            features.append(url.count('www'))

            # 20. Số lần xuất hiện '.com'
            features.append(url.count('.com'))

            # 21. HTTPS trong URL
            features.append(1 if 'https' in url else 0)

            # 22. Tỉ lệ ký tự số trong URL
            digits = sum(c.isdigit() for c in url)
            features.append(digits / len(url) if len(url) > 0 else 0)

            # 23. Tỉ lệ ký tự số trong hostname
            digits_in_host = sum(c.isdigit() for c in hostname)
            features.append(digits_in_host / len(hostname) if len(hostname) > 0 else 0)

            # 24. Số lượng subdomain
            subdomain_count = len(hostname.split('.')) - 2 if hostname else 0
            features.append(subdomain_count)

            # 25. Độ dài từ ngắn nhất trong hostname
            words_in_host = re.split(r'[\.-]', hostname)
            shortest_word_host = min([len(word) for word in words_in_host if word]) if words_in_host else 0
            features.append(shortest_word_host)

            # 26. Độ dài từ dài nhất trong hostname
            longest_word_host = max([len(word) for word in words_in_host if word]) if words_in_host else 0
            features.append(longest_word_host)

            # 27. Độ dài từ trung bình trong hostname
            avg_word_host = sum([len(word) for word in words_in_host]) / len(words_in_host) if words_in_host else 0
            features.append(avg_word_host)

            # 28. URL sử dụng tên miền rút gọn
            shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 'is.gd']
            features.append(1 if any(shortener in url for shortener in shorteners) else 0)

            # 29. URL chứa địa chỉ email
            features.append(1 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', url) else 0)

            # 30. Tỉ lệ ký tự chữ cái trong URL
            letters = sum(c.isalpha() for c in url)
            features.append(letters / len(url) if len(url) > 0 else 0)

            # 31. Số lượng từ trong URL
            words_in_url = len(re.findall(r'\w+', url))
            features.append(words_in_url)

            # 32. Độ dài của phần path
            path = parsed_url.path
            features.append(len(path))

            # 33. Số lượng ký tự trong path
            features.append(len(re.findall(r'\w+', path)))

            # 34. URL chứa domain đáng ngờ
            suspicious_domains = ['login', 'secure', 'account', 'verify', 'update',
                                  'authentication', 'confirmation', 'validation',
                                  'security-check', 'identity-verify']
            features.append(1 if any(domain in url for domain in suspicious_domains) else 0)

            # 35. Tỉ lệ ký tự số trong path
            digits_in_path = sum(c.isdigit() for c in path)
            features.append(digits_in_path / len(path) if len(path) > 0 else 0)

            # 36. Số lượng dấu '/' trong path
            features.append(path.count('/'))

            # 37. URL sử dụng TLD không phổ biến
            uncommon_tlds = ['xyz', 'top', 'click', 'online', 'icu', 'info',
                             'tk', 'ga', 'cf', 'ml', 'gq', 'loan', 'win',
                             'shop', 'site', 'us', 'pw']
            tld = hostname.split('.')[-1] if hostname else ''
            features.append(1 if tld in uncommon_tlds else 0)

            # 38. Số lượng tham số trong query
            query = parsed_url.query
            features.append(len(query.split('&')) if query else 0)

            # 39. Độ dài của phần query
            features.append(len(query) if query else 0)

            # 40. Query có chứa ký tự đặc biệt đáng ngờ không
            suspicious_query_chars = ['%20', '%22', '%27', '%3C', '%3E']
            features.append(1 if any(char in query for char in suspicious_query_chars) else 0)

            # 41. Tên miền chứa thương hiệu phổ biến
            trusted_domains = ['facebook.com', 'google.com', 'paypal.com', 'amazon.com',
                               'microsoft.com', 'apple.com', 'linkedin.com', 'twitter.com',
                               'netflix.com', 'bank.com', 'creditcard.com']
            features.append(1 if any(
                brand in hostname and hostname not in trusted_domains
                for brand in trusted_domains
            ) else 0)

            # 42. Tổng độ dài các subdomain
            subdomains = hostname.split('.')[:-2] if hostname else []
            features.append(sum(len(sub) for sub in subdomains))

            # 43. Subdomain đáng ngờ
            suspicious_subdomains = ['login', 'secure', 'account', 'auth', 'verify',
                                     'security', 'validation', 'confirm', 'admin']
            features.append(1 if any(sub in subdomains for sub in suspicious_subdomains) else 0)

            # 44. Tỉ lệ dấu '/' trong URL
            features.append(url.count('/') / len(url) if len(url) > 0 else 0)

            # 45. Path chứa từ khóa đáng ngờ
            suspicious_path_keywords = ['reset', 'confirm', 'admin', 'auth',
                                        'login', 'account', 'password',
                                        'recover', 'verification', 'update-credentials']
            features.append(1 if any(keyword in path for keyword in suspicious_path_keywords) else 0)

            # 46. Hostname chứa ký tự đặc biệt
            features.append(1 if any(char in hostname for char in ['_', '~', '$']) else 0)

            # 47. Tỉ lệ từ ngắn bất thường trong hostname
            invalid_words_in_host = [word for word in words_in_host if len(word) < 3]
            features.append(len(invalid_words_in_host) / len(words_in_host) if words_in_host else 0)

            # 48. Số lượng dấu '.' trong hostname
            features.append(hostname.count('.'))

            # 49. HTTPS trong hostname nhưng URL không bắt đầu bằng HTTPS
            features.append(1 if 'https' in hostname and not url.startswith('https') else 0)

            # 50. Độ dài của TLD
            features.append(len(tld))

            # 51. Query kết thúc bằng ký tự không hợp lệ
            features.append(1 if re.search(r'=[&]*$', query) else 0)

            # 52. Dấu hiệu chuyển hướng đáng ngờ
            features.append(1 if 'http://' in query or 'http://' in path else 0)

            # 53. Kiểm tra URL có chứa từ khóa không tin cậy trong phần path
            untrusted_path_keywords = ['fake', 'fraud', 'phishing', 'malware']
            features.append(1 if any(keyword in path for keyword in untrusted_path_keywords) else 0)

            # 54. Số lượng ký tự hoa trong URL (Kiểm tra xem URL có sử dụng nhiều chữ cái in hoa không)
            uppercase_count = sum(1 for c in url if c.isupper())
            features.append(uppercase_count)

            # 55. Tỉ lệ ký tự in hoa so với tổng ký tự trong URL
            features.append(uppercase_count / len(url) if len(url) > 0 else 0)

            # 56. Độ dài phần fragment (phần sau dấu '#')
            fragment = parsed_url.fragment
            features.append(len(fragment) if fragment else 0)

            # 57. URL có chứa từ khóa đáng ngờ trong fragment không
            suspicious_fragment_keywords = ['reset', 'confirm', 'secure', 'auth']
            features.append(1 if any(keyword in fragment for keyword in suspicious_fragment_keywords) else 0)

            # 58. Mã hóa đáng ngờ
            suspicious_encodings = ['%00', '%3A', '%2F', '%2E', '%5C']
            features.append(1 if any(encoding in url for encoding in suspicious_encodings) else 0)

            # 59. Ký tự lặp lại bất thường
            features.append(1 if re.search(r'\.\.\.\.|----', url) else 0)

            # 60. Tên miền quốc gia không phổ biến
            uncommon_ccTLDs = ['.tk', '.ga', '.cf', '.ml', '.gq']
            features.append(1 if any(tld.endswith(ccTLD) for ccTLD in uncommon_ccTLDs) else 0)

            # 61. Ký tự đặc biệt bất thường trong hostname
            special_chars_in_host = ['%', '$', '^', '&', '*', '(', ')', '=', '+', '#']
            features.append(sum(hostname.count(char) for char in special_chars_in_host))

            # 62. Từ khóa bảo mật trong bối cảnh đáng ngờ
            suspicious_security_keywords = ['secure', 'ssl', 'certified']
            features.append(1 if any(
                keyword in hostname and hostname not in trusted_domains
                for keyword in suspicious_security_keywords
            ) else 0)

            # 63. Độ dài URL bất thường
            features.append(1 if len(url) < 15 or len(url) > 200 else 0)

            # 64. Chuỗi giống mã trong hostname
            random_string_pattern = r'[a-zA-Z0-9]{10,}'
            features.append(1 if re.search(random_string_pattern, hostname) else 0)

        else:
            features = [0] * 64  # Nếu URL không hợp lệ, điền toàn bộ bằng 0

        return features

    def predict(self, url):
        # Rút trích đặc trưng và dự đoán xem URL có phải phishing không
        features = self.extract_features(url)
        features_df = pd.DataFrame([features])  # Đưa đặc trưng vào DataFrame
        prediction = self.model.predict(features_df)  # Dự đoán bằng mô hình
        return bool(prediction[0])  # Trả về True nếu là phishing


class MessageAnalyzer:
    def __init__(self):
        # Khởi tạo mô hình transformer để phân tích URL
        self.url_extractor = URLTransformerExtractor()
        self.url_extractor.eval()  # Đặt mô hình ở chế độ đánh giá (eval mode)
        # Khởi tạo mô hình phát hiện phishing
        self.phishing_detector = PhishingDetector()

        # Thêm pattern URL regex
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/\S*)?'
        )

    def extract_urls_with_transformer(self, message):
        """Sử dụng kết hợp transformer và regex để phát hiện URL"""
        urls = []

        # Sử dụng regex để tìm các URL tiềm năng
        potential_urls = self.url_pattern.finditer(message)

        for match in potential_urls:
            url = match.group()
            if self._is_valid_url(url):
                # Nếu URL không bắt đầu bằng http/https, thêm vào
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                urls.append(url)
                print(f"Detected URL: {url}")

        # Sử dụng transformer để xác nhận và phân loại URL
        if urls:
            with torch.no_grad():   # Không tính toán gradient để tiết kiệm bộ nhớ
                for url in urls:
                    url_logits = self.url_extractor(url)
                    url_probs = torch.softmax(url_logits, dim=-1)

                    # Nếu transformer xác nhận đây là URL (probability > 0.3)
                    if url_probs[0, 0, 1] > 0.3:
                        print(f"Transformer confirmed URL: {url} with probability {url_probs[0, 0, 1]}")
                    else:
                        urls.remove(url)
                        print(f"Transformer rejected URL: {url}")

        return urls

    def _is_valid_url(self, url):
        """Cải thiện kiểm tra URL hợp lệ"""
        try:
            # Nếu URL không có scheme, thêm https:// tạm thời để parse
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            result = urlparse(url)

            # Kiểm tra các điều kiện bổ sung
            conditions = [
                bool(result.scheme in ['http', 'https']),  # Kiểm tra xem có phải http hoặc https không
                bool(result.netloc),  # Kiểm tra có domain không
                len(result.netloc.split('.')) >= 2,  # Kiểm tra domain có ít nhất 2 phần
                not result.netloc.startswith('.'),  # Không bắt đầu bằng dấu chấm
                not result.netloc.endswith('.'),  # Không kết thúc bằng dấu chấm
            ]

            return all(conditions)

        except Exception as e:
            print(f"URL validation error: {str(e)}")
            return False

    def analyze_message(self, message):
        """Phân tích tin nhắn để tìm và kiểm tra URLs"""
        # Sử dụng transformer để trích xuất URL
        urls = self.extract_urls_with_transformer(message)

        # Sử dụng model pkl để phát hiện phishing
        phishing_urls = [url for url in urls if self.phishing_detector.predict(url)]
        # In ra kết quả ở console
        status = 'Phishing' if phishing_urls else 'Legit'
        print(f"Status: {status}")

        return {
            'has_urls': bool(urls),  # Kiểm tra xem có URL nào không
            'urls': urls,  # Trả về danh sách URL tìm được
            'phishing_detected': bool(phishing_urls),  # Kiểm tra có URL phishing không
            'phishing_urls': phishing_urls  # Trả về các URL phishing phát hiện
        }


# Khởi tạo analyzer
message_analyzer = MessageAnalyzer()

# Initialize the user manager
user_manager = user_manager.UserManager()


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/chat')
def chat():
    if 'username' not in session:
        return render_template('login.html')
    username = session['username']
    active_users = user_manager.get_active_users()
    return render_template('chat.html', username=username, active_users=active_users)


@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form.get('username')

    if not username:
        return jsonify({'success': False, 'message': 'Tên đăng nhập không hợp lệ.'})

    # Generate a unique session ID
    session_id = os.urandom(16).hex()

    # Try to add the user
    if user_manager.add_user(username, session_id):
        session['username'] = username
        session['session_id'] = session_id
        # Broadcast to all clients that a new user has joined
        socketio.emit('user_joined', {'username': username, 'active_users': list(user_manager.get_active_users())})
        return jsonify({'success': True, 'redirect': '/chat'})
    else:
        return jsonify({'success': False, 'message': 'Tên đăng nhập đã được sử dụng. Vui lòng chọn tên khác.'})


@app.route('/logout', methods=['POST'])
def logout():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Không tìm thấy phiên đăng nhập.'})

    username = session['username']
    session_id = session.get('session_id')

    # Remove user from active users
    if user_manager.remove_user(username=username):
        # Clear session
        session.pop('username', None)
        session.pop('session_id', None)

        # Notify all clients about user logout
        socketio.emit('user_logout', {
            'username': username,
            'active_users': list(user_manager.get_active_users())
        })
        return jsonify({'success': True, 'redirect': '/'})

    return jsonify({'success': False, 'message': 'Không thể đăng xuất. Vui lòng thử lại.'})


@socketio.on('connect')
def handle_connect():
    if 'username' in session:
        username = session['username']
        emit('user_joined', {
            'username': username,
            'active_users': list(user_manager.get_active_users())
        }, broadcast=True)


@socketio.on('disconnect')
def handle_disconnect():
    if 'username' in session:
        username = session['username']
        # Handle unexpected disconnections (browser close, etc.)
        if user_manager.remove_user(username=username):
            socketio.emit('user_logout', {
                'username': username,
                'active_users': list(user_manager.get_active_users())
            })


@socketio.on('send_message')
def handle_message(data):
    try:
        # Ensure only logged-in users can send messages
        if 'username' not in session:
            emit('error', {'message': 'Vui lòng đăng nhập trước'})
            return

        sender = session['username']
        message = data['message']

        # Phân tích tin nhắn using existing message_analyzer
        analysis = message_analyzer.analyze_message(message)

        # Chuẩn bị phản hồi
        response = {
            'sender': sender,
            'message': message,
            'phishing_detected': analysis['phishing_detected'],
            'phishing_urls': analysis['phishing_urls'],
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

        if analysis['phishing_detected']:
            response['warning'] = "Cảnh báo: Phát hiện URL lừa đảo trong tin nhắn!"

        # Gửi phản hồi qua WebSocket
        emit('receive_message', response, broadcast=True)

    except Exception as e:
        print(f"Error handling message: {str(e)}")
        emit('error', {'message': 'Có lỗi xảy ra khi xử lý tin nhắn'})


if __name__ == '__main__':
    socketio.run(app, debug=True)