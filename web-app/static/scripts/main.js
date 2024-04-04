function chooseFile() {
  document.getElementById("file").click();
}

const submitBtn = document.getElementById('submitBtn');

submitBtn.addEventListener("click", active);

function active() {
submitBtn.classList.toggle("is_active");
}

const fileStuff = document.querySelector("#file");

fileStuff.addEventListener("input", () => {
  const fileName = fileStuff.files[0]?.name;
  const fileNameLabel= document.querySelector("#fileName");
  fileNameLabel.textContent = fileName ?? "Blud";
});
