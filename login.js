async function establishServerSession(firebaseUser) {
      const idToken = await firebaseUser.getIdToken();
      const res = await fetch('/api/auth/firebase-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ idToken })
      });
      const data = await res.json().catch(() => ({}));
      return { res, data };
    }
    document.querySelector(".login").addEventListener("click", async () => {
      const email = document.getElementById("email").value.trim();
      const password = document.getElementById("password").value;
      if (!email || !password) { alert("⚠️ Please fill in both email and password."); return; }
      if (!email.includes("@")) { alert("⚠️ Invalid email format."); return; }
      try {
        const cred = await window._fb.signInWithEmailAndPassword(window._auth, email, password);
        const { res, data } = await establishServerSession(cred.user);
        if (res.ok && data.ok) window.location.href = "dashboard.html";
        else alert("⚠️ Login failed: " + (data.error || res.statusText));
      } catch (e) {
        alert("⚠️ Login failed: " + (e.message || "unknown error"));
      }
    });
    document.getElementById("google-login").addEventListener("click", async () => {
  try {
    await window._fb.signInWithRedirect(window._auth, new window._fb.GoogleAuthProvider());
    // page navigates away here; nothing after this line runs until the user returns
  } catch (e) {
    alert("⚠️ Sign-in failed: " + (e.message || "unknown error"));
  }
});
document.getElementById("github-login").addEventListener("click", async () => {
  try {
    await window._fb.signInWithRedirect(window._auth, new window._fb.GithubAuthProvider());
  } catch (e) {
    alert("⚠️ Sign-in failed: " + (e.message || "unknown error"));
  }
});
(async () => {
  try {
    const cred = await window._fb.getRedirectResult(window._auth);
    if (cred && cred.user) {
      const { res, data } = await establishServerSession(cred.user);
      if (res.ok && data.ok) window.location.href = "dashboard.html";
      else alert("⚠️ Sign-in failed: " + (data.error || res.statusText));
    }
  } catch (e) {
    if (e && e.code && e.code !== 'auth/null-user') {
      alert("⚠️ Sign-in failed: " + (e.message || "unknown error"));
    }
  }
})();
