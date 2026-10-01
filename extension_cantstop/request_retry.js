// Bounded recovery for transient transport failures; never retry invalid input.
async function advisorWithRetry(operation, {
  isCurrent = () => true, onRetry = () => {}, onError = () => {},
  wait = ms => new Promise(resolve => setTimeout(resolve, ms)),
  attempts = 3
} = {}) {
  for (let attempt = 1; attempt <= attempts; attempt++) {
    if (!isCurrent()) throw Object.assign(new Error("Position changed"), {cancelled:true});
    try {
      return await operation();
    } catch (error) {
      if (!isCurrent()) throw Object.assign(new Error("Position changed"), {cancelled:true});
      onError(error, attempt);
      const transient = error.status === 0 || [408,429,500,502,503,504].includes(error.status);
      if (!transient || attempt === attempts) throw error;
      onRetry(attempt);
      await wait(attempt === 1 ? 500 : 1500);
    }
  }
}
