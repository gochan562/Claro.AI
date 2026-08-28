const fileInput = document.getElementById("modelFile");
    const fileNameDisplay = document.getElementById("fileName");

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        fileNameDisplay.textContent = `Selected: ${file.name}`;
      } else {
        fileNameDisplay.textContent = "";
      }
    });

document.getElementById('nav-login').addEventListener('click', () => {
  location.href = 'Login.html';
});
document.getElementById('nav-signup').addEventListener('click', () => {
  location.href = 'signup.html';
});
