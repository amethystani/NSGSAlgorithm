#!/bin/bash

# Start Backend Server
osascript -e 'tell app "Terminal" to do script "cd '$PWD'/Backend && npm start"'

# Wait a bit for backend to initialize
sleep 2

# Start Frontend Server
osascript -e 'tell app "Terminal" to do script "cd '$PWD'/Frontend && npx expo start"'

echo "Servers started in separate terminal windows!" 