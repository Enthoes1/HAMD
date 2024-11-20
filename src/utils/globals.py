from flask_socketio import SocketIO

# 创建一个全局的 SocketIO 实例
socketio = SocketIO()

def init_socketio(app):
    """初始化 socketio"""
    global socketio
    socketio.init_app(app, async_mode='eventlet', cors_allowed_origins="*")
    return socketio 