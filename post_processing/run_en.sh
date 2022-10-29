# EN
docker run -e CONFIG=brillmoore_wikitypos_en -p 8082:5000 \
    --runtime=nvidia \
    -v model_data_en:/root/.deeppavlov \
    -v venv:/venv \
    --name "spellchecking_en" \
    deeppavlov:latest