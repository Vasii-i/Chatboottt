// assets/main.js
window.confettiLike = function() {
  if (document.getElementById('confetti-canvas')) return;
  var c = document.createElement('canvas');
  c.id = 'confetti-canvas';
  Object.assign(c.style, {
    position:'fixed', top:0, left:0, width:'100vw', height:'100vh',
    pointerEvents:'none', zIndex:9999
  });
  document.body.appendChild(c);
  var ctx = c.getContext('2d');
  var W = window.innerWidth, H = window.innerHeight;
  c.width = W; c.height = H;

  var n = 130;
  var pieces = Array.from({length:n}, () => ({
    x: Math.random()*W, y: Math.random()*H*0.4,
    r: 6 + Math.random()*8, d: 2 + Math.random()*2,
    c: `hsl(${Math.random()*360},90%,78%)`, t: Math.random()*Math.PI*2
  }));

  function draw(){
    ctx.clearRect(0,0,W,H);
    for (var i=0;i<pieces.length;i++){
      var p = pieces[i];
      ctx.beginPath();
      ctx.arc(p.x,p.y,p.r,0,2*Math.PI);
      ctx.fillStyle=p.c;
      ctx.globalAlpha=0.85;
      ctx.fill();
    }
    update();
    requestAnimationFrame(draw);
  }
  function update(){
    for (var i=0;i<pieces.length;i++){
      var p = pieces[i];
      p.y += p.d; p.x += Math.sin(p.t);
      if (p.y > H) { p.y = -10; p.x = Math.random()*W; }
    }
  }
  draw();
  setTimeout(()=>{ c.remove(); }, 1800);
};
