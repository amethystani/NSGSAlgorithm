// Static files
app.use(express.static('public'));
app.use('/uploads', express.static('Imginput'));

// Add this new route to serve processed images
app.use('/processed', express.static('Imgoutput')); 

console.log(`C++ processing completed in ${processingTime.toFixed(2)} seconds`);
console.log(`Command output: ${stdout.substring(0, 500)}${stdout.length > 500 ? '...' : ''}`);

// Capture the NSGS logs from stdout
let nsgsLogs = stdout || "";

// Add diagnostic logging to help troubleshoot
console.log(`NSGS logs available: ${nsgsLogs.length > 0 ? 'Yes' : 'No'} (length: ${nsgsLogs.length})`);
if (nsgsLogs.length > 0) {
  console.log(`NSGS logs sample: ${nsgsLogs.substring(0, 200)}${nsgsLogs.length > 200 ? '...' : ''}`);
} else {
  console.log("WARNING: No NSGS logs captured from stdout");
  nsgsLogs = "No logs were generated from the NSGS process. This might indicate an issue with the process execution.";
}

// Parse some basic metrics from the NSGS logs 

// After the nsgsStats object is defined
const nsgsStats = {
  graphNodes,
  processedSpikes,
  queueSize,
  adaptationMultiplier: 1.0,
  processingTime: detectionTime || processingTime,
  status,
  logsOutput: nsgsLogs // Include the full logs (potentially empty)
};

// Store the last NSGS logs for debugging
app.locals.lastNsgsLogs = nsgsLogs;
app.locals.lastNsgsStats = nsgsStats;

// Add a debug endpoint to retrieve NSGS logs
app.get('/api/debug/nsgs-logs', (req, res) => {
  const lastLogs = app.locals.lastNsgsLogs || 'No logs available yet';
  const lastStats = app.locals.lastNsgsStats || { status: 'No stats available yet' };
  
  res.json({
    logs: lastLogs,
    stats: lastStats,
    timestamp: new Date().toISOString()
  });
}); 