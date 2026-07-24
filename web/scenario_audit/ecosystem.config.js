// pm2 config for the SCENARIO construct-audit server (sibling of web/app "whosaw").
// Start:  cd /path/to/launderbench/web/scenario_audit && pm2 start ecosystem.config.js
module.exports = {
  apps: [{
    name: "scenaudit",
    script: "server.py",
    interpreter: "python3",
    cwd: "/path/to/launderbench/web/scenario_audit",
    env: {
      PORT: "8093",
      BASE_PATH: "/scenaudit/",  // public path behind nginx
      COOKIE_SECURE: "1"         // requires HTTPS
    },
    autorestart: true,
    max_restarts: 10
  }]
};
