// Toolbar button: show / hide the overlay in the current tab.
chrome.action.onClicked.addListener(tab => {
  if (tab.id !== undefined) chrome.tabs.sendMessage(tab.id, { type: 'toggle' }).catch(() => {});
});
