const text = "Drag. Drop. Deploy.";
    const target = document.getElementById("typewriter");
    let i = 0;
    const speed = 80;
    function type() {
      if (i < text.length) {
        target.innerHTML += text.charAt(i);
        i++;
        setTimeout(type, speed);
      }
    }
    type();
