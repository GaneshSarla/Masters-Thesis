function updateFileName() {
    // Get the file input element
    var fileInput = document.getElementById('file-upload');

    // Get the selected file name
    var fileName = fileInput.files[0].name;

    // Update the file name display
    document.getElementById('file-name').textContent = fileName;

}
// Function to update the file name on selection
document.getElementById('file-upload').addEventListener('change', function () {
    const fileName = this.files[0] ? this.files[0].name : '';
    document.getElementById('file-name').textContent = fileName;
});

// Form validation function
function validateForm() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];

    // Check if no file is selected
    if (!file) {
        alert('Please select a file to upload.');
        return false; // Prevent form submission
    }

    // Check if the file is a CSV
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (fileExtension !== 'csv') {
        alert('Please upload a valid CSV file.');
        return false; // Prevent form submission
    }

    // If everything is fine, allow form submission
    return true;
}

