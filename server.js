// Static files
app.use(express.static('public'));
app.use('/uploads', express.static('Imginput'));

// Add this new route to serve processed images
app.use('/processed', express.static('Imgoutput')); 