const browserAPI = typeof browser !== "undefined" ? browser : chrome;

document.getElementById("captureBtn").addEventListener("click", function() {
  console.log("Step 1: Button was clicked");

  browserAPI.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    console.log("Step 2: Found tabs", tabs);

    if (!tabs || tabs.length === 0) {
      document.getElementById("status").textContent = "Error: no active tab found";
      return;
    }

    console.log("Step 3: Sending message to tab", tabs[0].id);

    browserAPI.tabs.sendMessage(tabs[0].id, { action: "capture" }, function(response) {
      console.log("Step 4: Got response", response);
      document.getElementById("status").textContent = "State captured!";
    });

// When popup opens, show how many states we've captured so far
browserAPI.storage.local.get("captured_states", function(result) {
  const states = result.captured_states || [];
  document.getElementById("status").textContent = "Saved states: " + states.length;
});
  });
});