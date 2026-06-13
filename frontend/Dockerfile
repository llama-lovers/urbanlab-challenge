# Next.js frontend (dev mode — matches the bind-mount workflow of the other services).
FROM node:22-alpine

WORKDIR /app

# Install deps first for better layer caching. The container's node_modules is kept
# in an anonymous volume (see docker-compose) so it isn't shadowed by the bind mount.
COPY package.json package-lock.json* ./
RUN npm install

COPY . .

EXPOSE 3000

# Bind to 0.0.0.0 so the published port is reachable from the host.
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
