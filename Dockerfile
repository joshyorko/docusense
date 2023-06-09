# Use the official Redis image 
FROM redis:latest

# Expose port 6379
EXPOSE 6379

# Run Redis without authentication
CMD ["redis-server"]  