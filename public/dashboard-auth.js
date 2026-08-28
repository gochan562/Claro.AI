(async function checkAuth(){
      try {
        const res = await fetch('/api/auth/me', { credentials: 'include' });
        if (!res.ok) { window.location.href = 'Login.html'; return; }
        const data = await res.json();
        const user = data.user || {};
        const userNameElement = document.getElementById('user-name');
        const userAvatarElement = document.querySelector('.user img');
        if (userNameElement) userNameElement.textContent = user.username || user.email || 'User';
        if (userAvatarElement) {
          const seed = user.email || user.username || 'user';
          userAvatarElement.src = `https://api.dicebear.com/7.x/shapes/svg?seed=${encodeURIComponent(seed)}`;
        }
      } catch (e) {
        window.location.href = 'Login.html';
      }
    })();
    window.logout = async function() {
      try {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
      } catch (e) {}
      window.location.href = 'Login.html';
    };
