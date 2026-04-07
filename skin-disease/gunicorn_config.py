
# Gunicorn configuration for Skin Disease Classifier
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"

# Process naming
proc_name = "skin-disease-classifier"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None

# SSL
keyfile = None
certfile = None

# Application
preload_app = False
paste = None
