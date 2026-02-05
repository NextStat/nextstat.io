# NextStat Governance

> Документ описывает процесс управления проектом NextStat, роли участников и процедуры принятия решений.

## Принципы

1. **Открытость** — все решения принимаются публично в Issues и PR
2. **Меритократия** — влияние основано на качестве и количестве вклада
3. **Консенсус** — стремимся к согласию, но имеем процедуру разрешения конфликтов
4. **Инклюзивность** — приветствуем участников с любым уровнем опыта

## Роли

### Contributor

Любой человек, внесший хотя бы один PR.

**Права:**
- Создавать Issues
- Отправлять Pull Requests
- Комментировать Issues и PR
- Участвовать в обсуждениях

**Обязанности:**
- Следовать [Code of Conduct](#code-of-conduct)
- Подписывать коммиты DCO (см. [DCO.md](DCO.md))
- Следовать [CONTRIBUTING.md](CONTRIBUTING.md)

### Committer

Contributor, продемонстрировавший постоянный высококачественный вклад.

**Критерии:**
- Минимум 10 merged PR
- Активность в течение последних 3 месяцев
- Понимание архитектуры проекта
- Одобрение минимум 2 Maintainers

**Права:**
- Все права Contributor
- Review и approve PR других участников
- Триаж Issues (assign, label, milestone)
- Участие в техническом голосовании

**Обязанности:**
- Review PR (стремиться к < 48 часов для первого feedback)
- Участие в release процессе
- Поддержка новых contributors

**Назначение:**
1. Maintainer создает Issue с предложением
2. Голосование Maintainers (72 часа)
3. Требуется минимум 2 "за" и нет "против"

### Maintainer

Ключевые участники проекта с полными правами.

**Критерии:**
- Committer минимум 6 месяцев
- Глубокое понимание всей кодовой базы
- Демонстрация лидерских качеств
- Одобрение большинства текущих Maintainers

**Права:**
- Все права Committer
- Merge Pull Requests
- Управление релизами
- Управление ролями (назначение Committers)
- Финальное решение в технических спорах

**Обязанности:**
- Обеспечение качества кода
- Поддержка долгосрочного видения проекта
- Менторинг Committers и Contributors
- Участие в стратегическом планировании

**Текущие Maintainers:**
- @andresvlc (Project Lead)

**Назначение:**
1. Любой Maintainer может предложить кандидата
2. Обсуждение среди Maintainers (1 неделя)
3. Голосование (требуется 2/3 голосов "за")

### Project Lead

Основатель проекта, финальная инстанция в сложных вопросах.

**Права:**
- Все права Maintainer
- Финальное решение в конфликтах
- Изменение governance процесса

**Обязанности:**
- Долгосрочное видение проекта
- Разрешение блокирующих конфликтов
- Представление проекта во внешнем мире

**Текущий Project Lead:** @andresvlc

## Процесс принятия решений

### Типы решений

#### 1. Повседневные (Day-to-day)

**Примеры:** Bug fixes, документация, мелкие улучшения

**Процесс:**
1. Contributor создает PR
2. Минимум 1 Committer/Maintainer делает review
3. После approve PR может быть merged

**Требования:**
- Все тесты проходят
- DCO sign-off на всех коммитах
- Следование coding standards

#### 2. Значительные (Significant)

**Примеры:** Новые возможности, рефакторинг модулей, изменения API

**Процесс:**
1. Создать Issue для обсуждения перед реализацией
2. Обсуждение (минимум 48 часов)
3. Если есть консенсус → создать PR
4. Review минимум 2 Committers/Maintainers
5. Merge после approve

#### 3. Критические (Critical)

**Примеры:** Архитектурные изменения, breaking changes, изменение лицензии

**Процесс:**
1. Создать RFC (Request for Comments) в `docs/rfcs/`
2. Публичное обсуждение (минимум 2 недели)
3. Голосование Maintainers (требуется 2/3 "за")
4. Реализация только после одобрения

**Формат RFC:**
```markdown
# RFC-0001: Title (Template)

## Summary
Brief description

## Motivation
Why we need this

## Detailed Design
Technical details

## Drawbacks
Known issues

## Alternatives
What else was considered

## Unresolved Questions
Open questions
```

### Разрешение конфликтов

1. **Обсуждение** — попытка достичь консенсуса в Issue/PR
2. **Escalation** — если консенсус не достигнут, escalate к Maintainers
3. **Голосование** — Maintainers голосуют (простое большинство)
4. **Финальное решение** — Project Lead разрешает блокирующие конфликты

## Процесс Review

### Для Reviewers

**Что проверять:**
- [ ] Тесты проходят и покрывают новый код
- [ ] Код следует style guide (rustfmt, clippy)
- [ ] DCO sign-off на всех коммитах
- [ ] Документация обновлена (если нужно)
- [ ] Нет breaking changes без RFC
- [ ] Производительность не ухудшилась
- [ ] Безопасность (нет SQL injection, XSS, etc.)

**Тон review:**
- Конструктивный и дружелюбный
- Объясняйте "почему", не только "что"
- Признавайте хорошую работу

**Timeline:**
- Первый feedback: < 48 часов
- Полный review: < 1 неделя

### Для Authors

**После получения review:**
1. Ответить на все комментарии
2. Внести изменения или объяснить почему нет
3. Запросить re-review
4. Быть терпеливым и уважительным

## Release Process

### Версионирование

Следуем [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0) — breaking changes
- **MINOR** (0.X.0) — новые возможности (backward compatible)
- **PATCH** (0.0.X) — bug fixes

### Release Schedule

- **Patch releases** — по мере необходимости (critical bugs)
- **Minor releases** — каждые 6-8 недель
- **Major releases** — когда необходимо (breaking changes накопились)

### Release Procedure

1. Maintainer создает Issue "Release vX.Y.Z"
2. Prepare:
   - Update CHANGELOG.md
   - Update version in Cargo.toml
   - Run full test suite
   - Build documentation
3. Create release branch: `release/vX.Y.Z`
4. Tag: `git tag -s vX.Y.Z`
5. Push tag → triggers CI/CD → publish to crates.io and PyPI
6. Create GitHub Release with notes
7. Announce on community channels

## Изменение Governance

Этот документ может быть изменен через RFC процесс:

1. Создать RFC с предложением изменений
2. Публичное обсуждение (минимум 4 недели)
3. Голосование всех Maintainers (требуется 3/4 "за")
4. Project Lead имеет veto право

## Code of Conduct

Мы стремимся создать дружелюбное сообщество:

- **Будьте уважительны** к другим участникам
- **Будьте конструктивны** в критике
- **Будьте терпеливы** с новичками
- **Не допускается** harassment, discrimination, trolling

Нарушения могут привести к:
1. Warning
2. Temporary ban (1-4 недели)
3. Permanent ban

Сообщать о нарушениях: conduct@nextstat.io

## Контакты

- **General:** dev@nextstat.io
- **Governance вопросы:** governance@nextstat.io
- **Code of Conduct:** conduct@nextstat.io
- **Private (Maintainers):** maintainers@nextstat.io

---

*Последнее обновление: 2026-02-05*
