<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blood Cell Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            background-color: #f4f6f9; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 30px;
            max-width: 500px;
            width: 100%;
        }
        #result {
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mb-4">Blood Cell Classifier</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="fileUpload" class="form-label">Upload Blood Cell Image</label>
                <input class="form-control" type="file" id="fileUpload" name="file" accept="image/*" required>
            </div>
            <div class="text-center">
                <button type="submit" class="btn btn-primary">Classify Image</button>
            </div>
        </form>
        <div id="result" class="mt-4"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const resultDiv = document.getElementById('result');
            const submitButton = this.querySelector('button[type="submit"]');
            
            // Disable submit button and show loading
            submitButton.disabled = true;
            resultDiv.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                // Check if response is ok (status in 200-299 range)
                if (!response.ok) {
                    // Try to parse error response
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Re-enable submit button
                submitButton.disabled = false;
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    resultDiv.innerHTML = `
                        <div class="alert alert-success">
                            <strong>Predicted Class:</strong> ${data.class}<br>
                            <strong>Confidence:</strong> ${data.confidence}
                        </div>
                    `;
                }
            })
            .catch(error => {
                // Re-enable submit button
                submitButton.disabled = false;
                resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    </script>
</body>
</html>
