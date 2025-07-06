#!/bin/bash

# Быстрый скрипт для создания patch-релиза
# Автоматически увеличивает patch версию и создает релиз

set -e

# Цвета
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}⚡ Быстрый релиз Synapto Catcher${NC}"
echo "================================"

# Проверяем git репозиторий
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Не найден git репозиторий${NC}"
    exit 1
fi

# Коммитим изменения если есть
if ! git diff-index --quiet HEAD --; then
    echo -e "${BLUE}📝 Коммитим изменения...${NC}"
    git add .
    git commit -m "Prepare release version"
fi

# Получаем последнюю версию
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo "Текущая версия: $LAST_TAG"

# Увеличиваем patch версию
if [[ $LAST_TAG =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    MAJOR=${BASH_REMATCH[1]}
    MINOR=${BASH_REMATCH[2]}
    PATCH=${BASH_REMATCH[3]}
    NEW_VERSION="v$MAJOR.$MINOR.$((PATCH + 1))"
else
    NEW_VERSION="v1.0.0"
fi

echo -e "${GREEN}Новая версия: $NEW_VERSION${NC}"

# Создаем релиз
git tag -a "$NEW_VERSION" -m "Release version $NEW_VERSION"
git push origin main
git push origin "$NEW_VERSION"

echo -e "${GREEN}🎉 Релиз $NEW_VERSION создан!${NC}"
echo "Проверьте GitHub Actions для сборки EXE файла." 