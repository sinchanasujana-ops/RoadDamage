(function () {
  var dz = document.getElementById("dropzone");
  var input = document.getElementById("file-input");
  var text = document.getElementById("dz-text");
  var preview = document.getElementById("preview");
  var placeholder = document.getElementById("dz-placeholder");
  var conf = document.getElementById("conf");
  var confVal = document.getElementById("conf-val");
  var form = document.getElementById("scan-form");
  var btn = document.getElementById("run-btn");

  function showFile(file) {
    if (!file) return;
    text.innerHTML = 'Selected: <span class="fname">' + file.name + "</span>";
    var url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    if (placeholder) placeholder.style.display = "none";
  }

  ["dragenter", "dragover"].forEach(function (evt) {
    dz.addEventListener(evt, function (e) {
      e.preventDefault();
      dz.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dz.addEventListener(evt, function (e) {
      e.preventDefault();
      dz.classList.remove("drag");
    });
  });
  dz.addEventListener("drop", function (e) {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      showFile(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener("change", function () {
    if (input.files.length) showFile(input.files[0]);
  });
  if (conf && confVal) {
    conf.addEventListener("input", function () {
      confVal.textContent = Number(conf.value).toFixed(2);
    });
  }
  form.addEventListener("submit", function () {
    btn.classList.add("busy");
    btn.textContent = "Scanning…";
  });
})();
