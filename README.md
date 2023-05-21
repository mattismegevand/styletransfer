# styletransfer

Implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). \
Added [Total Variation loss](https://arxiv.org/abs/1412.0035) which was not implemented in the original paper.

| Content | Style | Result |
| :-----: |:-----:| :-----:|
| <img src="in/cat.jpg" width="256" height="256"> | <img src="in/matisse.jpg" width="256" height="256"> | <img src="out/gif/matisse.gif" width="256" height="256"> |
| <img src="in/cat.jpg" width="256" height="256"> | <img src="in/starry_night.jpg" width="256" height="256"> | <img src="out/gif/starry_night.gif" width="256" height="256"> |

<br/>

To transfer the style of an image, run the following commands:

```bash
./main.py cat.jpg matisse.jpg
```

Gif can also be created by running create_gif.py
