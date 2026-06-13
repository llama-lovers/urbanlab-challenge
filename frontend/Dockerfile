# Next.js frontend (dev mode — matches the bind-mount workflow of the other services).
FROM node:22-bookworm-slim

WORKDIR /app

# Upgrade npm: npm 10.x intermittently crashes with "Exit handler never called!"
# at the end of install on some hosts; npm 11 fixes it.
RUN npm install -g npm@11

# Install deps first for better layer caching. The container's node_modules is kept
# in an anonymous volume (see docker-compose) so it isn't shadowed by the bind mount.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .

EXPOSE 3000

# Bind to 0.0.0.0 so the published port is reachable from the host.
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
