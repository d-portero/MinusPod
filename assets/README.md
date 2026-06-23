# Assets Directory

Place your `replace.mp3` file here. This should be a 1-second beep/tone audio file that will replace advertisement segments in podcasts.

The file should be named exactly: `replace.mp3`

## Channel layout

Use a stereo (2-channel) file. MinusPod keeps each podcast's own channel layout
in the output, except when a replacement lands at the very start of an episode
(a pre-roll ad): there the spliced output takes the replacement's channel count
for that episode. So a mono `replace.mp3` can downmix a stereo show to mono,
while a stereo file is always safe (a mono show is adapted with no quality loss).
The bundled default is already stereo; this only matters if you supply your own.