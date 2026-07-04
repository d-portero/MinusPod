/**
 * Copy text to the clipboard. Returns true on success so callers only show
 * "Copied" when the copy actually happened.
 *
 * Falls back to a hidden input + execCommand when navigator.clipboard is
 * unavailable, which is common when MinusPod runs over plain http on a LAN
 * (clipboard API requires a secure context).
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const input = document.createElement('input');
      input.value = text;
      document.body.appendChild(input);
      input.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(input);
      return ok;
    } catch {
      return false;
    }
  }
}
