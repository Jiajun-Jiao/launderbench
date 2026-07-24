// pm2 config for the annotation server.
// Start:  cd /path/to/launderbench/web/app && pm2 start ecosystem.config.js
module.exports = {
  apps: [{
    name: "whosaw",
    script: "server.py",
    interpreter: "python3",
    cwd: "/path/to/launderbench/web/app",
    env: {
      PORT: "8092",
      BASE_PATH: "/whosaw/",   // public path behind nginx
      COOKIE_SECURE: "1"       // requires HTTPS (you have it)
    },
    autorestart: true,
    max_restarts: 10
  }]
};
