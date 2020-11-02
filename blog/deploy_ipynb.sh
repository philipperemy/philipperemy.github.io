FILENAME=$1
echo filename = $FILENAME
FILENAME_WO_EXT=${FILENAME%.*}

rm -rf $FILENAME_WO_EXT
ipython nbconvert $FILENAME --to markdown
mv ${FILENAME_WO_EXT}.md _posts
mkdir $FILENAME_WO_EXT
mv ${FILENAME_WO_EXT}_files $FILENAME_WO_EXT
rm -rf _site
