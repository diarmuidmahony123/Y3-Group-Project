document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.getElementById("login-form");
    const loginError = document.getElementById("login-error");
  
    document.addEventListener("DOMContentLoaded", function() {
        const loginForm = document.getElementById("login-form");
        const loginError = document.getElementById("login-error");
      
        loginForm.addEventListener("submit", async function(event) {
          event.preventDefault();
      
          const email = document.getElementById("email").value;
          const password = document.getElementById("password").value;
      
          try {
            const userCredential = await firebase.auth().signInWithEmailAndPassword(email, password);
            // You can redirect the user to another page or display a success message here
            console.log("User logged in:", userCredential.user.email);
          } catch (error) {
            // Handle login errors
            console.error("Login error:", error.message);
            loginError.textContent = "Invalid email or password. Please try again.";
          }
        });
      });    loginForm.addEventListener("submit", async function (event) {
      event.preventDefault();
  
      const email = document.getElementById("email").value;
      const password = document.getElementById("password").value;
  
      // You can implement your login logic here (e.g., check credentials against a database).
      // For this example, we'll simulate a successful login.
  
      if (email === "user@example.com" && password === "password") {
        // Redirect the user to the dashboard or home page upon successful login.
        window.location.href = "dashboard.html";
      } else {
        loginError.textContent = "Invalid email or password. Please try again.";
      }
    });
  });