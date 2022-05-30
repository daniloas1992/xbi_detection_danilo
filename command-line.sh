echo "===== Browserbite External ====="
date
python3 main.py browserbite external none >> results/browserbite-external.results.txt
echo "===== Browserbite Internal ====="
python3 main.py browserbite internal none >> results/browserbite-internal.results.txt

echo "===== CrossCheck External ====="
date
python3 main.py crosscheck external none >> results/crosscheck-external.results.txt
echo "===== CrossCheck Internal ====="
python3 main.py crosscheck internal none >> results/crosscheck-internal.results.txt

echo "===== BrowserNinja 1 External ====="
date
python3 main.py browserninja1 external none >> results/browserninja1-external.results.txt
echo "===== BrowserNinja 1 Internal ====="
python3 main.py browserninja1 internal none >> results/browserninja1-internal.results.txt

echo "===== CNN External ====="
date
python3 main.py image_diff_extractor external none >> results/image_diff_extractor-external.results.txt
echo "===== CNN Internal ====="
python3 main.py image_diff_extractor internal none >> results/image_diff_extractor-internal.results.txt

