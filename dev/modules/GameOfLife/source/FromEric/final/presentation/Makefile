DOC    := presentation
IMAGES := life1-cropped.png life2-cropped.png

include ../common/Makefile

%-cropped.png:%.png
	pngtopnm $+ | pnmcrop -margin 50 | pnmtopng > $@


