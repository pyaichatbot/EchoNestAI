FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY echo-nest-ai/package.json echo-nest-ai/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY echo-nest-ai/ .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:3000/api/health || exit 1

# Command to run the application
CMD ["npm", "start"]
