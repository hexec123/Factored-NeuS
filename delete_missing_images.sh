#!/bin/bash

DATASET="public_data/banc2"
IMAGES_DIR="$DATASET/image"
MASKS_DIR="$DATASET/mask"
COLMAP_IMAGES="$DATASET/sparse/0/images.txt"

# Vérifications
if [ ! -f "$COLMAP_IMAGES" ]; then
    echo "Erreur: $COLMAP_IMAGES introuvable"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Erreur: dossier images introuvable"
    exit 1
fi

echo "Lecture des images reconstruites..."

# Extraire les noms d'images reconstruites (ignorer les commentaires)
grep -v '^#' "$COLMAP_IMAGES" | awk 'NR % 2 == 1 {print $10}' > valid_images.txt

TOTAL_BEFORE=$(ls "$IMAGES_DIR" | wc -l)
REMOVED=0

echo "Suppression des images non reconstruites..."

for img in "$IMAGES_DIR"/*.png; do
    filename=$(basename "$img")
    if ! grep -qx "$filename" valid_images.txt; then
        rm "$img"
        if [ -f "$MASKS_DIR/$filename" ]; then
            rm "$MASKS_DIR/$filename"
        fi
        REMOVED=$((REMOVED + 1))
        echo "Supprimé: $filename"
    fi
done

TOTAL_AFTER=$(ls "$IMAGES_DIR" | wc -l)

rm valid_images.txt

echo "----------------------------------"
echo "Images avant: $TOTAL_BEFORE"
echo "Images supprimées: $REMOVED"
echo "Images restantes: $TOTAL_AFTER"
echo "Nettoyage terminé."