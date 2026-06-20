const browserAPI = typeof browser !== "undefined" ? browser : chrome;

const statusEl = document.getElementById("status");
const captureBtn = document.getElementById("captureBtn");

captureBtn.addEventListener("click", function () {
  statusEl.textContent = "Querying advisor...";

  browserAPI.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (!tabs || tabs.length === 0) {
      statusEl.textContent = "Error: no active tab found";
      return;
    }

    browserAPI.tabs.sendMessage(tabs[0].id, { action: "capture" }, function (response) {
      const err = browserAPI.runtime.lastError;
      if (err) {
        statusEl.textContent = "Error: " + err.message;
        return;
      }

      if (!response) {
        statusEl.textContent = "No response from content script";
        return;
      }

      if (response.ok) {
        const value = response.response && typeof response.response.value === "number" ? ` · ${(response.response.value * 100).toFixed(1)}%` : "";
        statusEl.textContent = "Advisor OK via " + response.transport + value;
      } else if (response.skipped) {
        statusEl.textContent = "Skipped: not a dice choice";
      } else {
        statusEl.textContent = "Error: " + response.error;
      }
    });
  });
});
