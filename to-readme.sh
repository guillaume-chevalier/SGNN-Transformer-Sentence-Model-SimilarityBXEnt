rm README.md

jupyter nbconvert --to python Train-SGNN-Transformer-Sentence-Model-SimilarityBXENT.ipynb
jupyter nbconvert --to python Load-and-Inspect-Model-Predictions.ipynb
jupyter nbconvert --to markdown Train-SGNN-Transformer-Sentence-Model-SimilarityBXENT.ipynb
jupyter nbconvert --to markdown Load-and-Inspect-Model-Predictions.ipynb

cat Train-SGNN-Transformer-Sentence-Model-SimilarityBXENT.md > README.md
cat Load-and-Inspect-Model-Predictions.md >> README.md

rm Train-SGNN-Transformer-Sentence-Model-SimilarityBXENT.md
rm Load-and-Inspect-Model-Predictions.md

