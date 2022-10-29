# RU
docker run -e CONFIG=levenshtein_corrector_ru -p 8081:5000 \
    --runtime=nvidia \
    -v model_data_ru:/root/.deeppavlov \
    -v venv:/venv \
    --rm \
    --name "spellchecking_ru" \
    deeppavlov:latest