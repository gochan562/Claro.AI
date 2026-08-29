async function establishServerSession(firebaseUser, username) {
      const idToken = await firebaseUser.getIdToken();
      const body = username ? { idToken, username } : { idToken };
      const res = await fetch('/api/auth/firebase-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });
      const data = await res.json().catch(() => ({}));
      return { res, data };
    }
    document.querySelector(".signup").addEventListener("click", async () => {
      const username = document.getElementById("username").value.trim();
      const email = document.getElementById("email").value.trim();
      const password = document.getElementById("password").value;
      if (!username || !email || !password) { alert("⚠️ Fill in all fields!"); return; }
      if (!email.includes("@")) { alert("⚠️ Invalid email format."); return; }
      if (password.length < 8) { alert("⚠️ Password must be at least 8 characters."); return; }
      try {
        const cred = await window._fb.createUserWithEmailAndPassword(window._auth, email, password);
        const { res, data } = await establishServerSession(cred.user, username);
        if (res.ok && data.ok) window.location.href = "dashboard.html";
        else alert("⚠️ Signup failed: " + (data.error || res.statusText));
      } catch (e) {
        alert("⚠️ Signup failed: " + (e.message || "unknown error"));
      }
    });
    document.getElementById("google-login").addEventListener("click", async () => {
  try {
    await window._fb.signInWithRedirect(window._auth, new window._fb.GoogleAuthProvider());
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
