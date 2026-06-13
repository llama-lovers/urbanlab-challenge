# Next.js frontend — production build.
FROM node:22-bookworm-slim

WORKDIR /app

# Upgrade npm: npm 10.x intermittently crashes with "Exit handler never called!"
# at the end of install on some hosts; npm 11 fixes it.
RUN npm install -g npm@11

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .

# NEXT_PUBLIC_* is inlined at build time, so the backend URL must be set here.
ARG NEXT_PUBLIC_API_URL=http://10.8.47.10:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

ENV NODE_ENV=production
EXPOSE 3000

# next start serves the built app (incl. /_next/static) — bind 0.0.0.0 for the LAN.
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
