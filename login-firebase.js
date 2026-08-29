import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
  import {
    getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword,
    GoogleAuthProvider, GithubAuthProvider, signInWithRedirect, getRedirectResult
  } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

  const firebaseConfig = {
    apiKey: "AIzaSyD9hhRlXX_PmTSlOnVAtRGtLwiGNTQPJV8",
    authDomain: "claro-ai-signup.firebaseapp.com",
    projectId: "claro-ai-signup",
    storageBucket: "claro-ai-signup.firebasestorage.app",
    messagingSenderId: "879903723540",
    appId: "1:879903723540:web:d7f90ec1c91e0ec27bd701"
  };
  const app = initializeApp(firebaseConfig);
  const auth = getAuth(app);
  window._auth = auth;
  window._fb = { signInWithEmailAndPassword, createUserWithEmailAndPassword,
                 GoogleAuthProvider, GithubAuthProvider, signInWithRedirect, getRedirectResult };
