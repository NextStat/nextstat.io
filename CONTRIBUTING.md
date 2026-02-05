# Руководство по вкладу в NextStat

Спасибо за интерес к NextStat! Мы рады любому вкладу — от исправления опечаток до новых возможностей.

## Оглавление

- [Кодекс поведения](#кодекс-поведения)
- [С чего начать](#с-чего-начать)
- [Процесс разработки](#процесс-разработки)
- [Требования к коду](#требования-к-коду)
- [Процесс pull request](#процесс-pull-request)
- [DCO Sign-off](#dco-sign-off)
- [Тестирование](#тестирование)
- [Документация](#документация)

## Кодекс поведения

Мы стремимся создать открытое и дружелюбное сообщество. Пожалуйста, будьте уважительны к другим участникам.

## С чего начать

### Найти задачу

1. Посмотрите [Issues](https://github.com/nextstat/nextstat/issues) с метками `good first issue` или `help wanted`
2. Прочитайте [docs/plans/README.md](docs/plans/README.md) для понимания архитектуры
3. Если у вас есть идея — сначала создайте Issue для обсуждения

### Настройка окружения

```bash
# 1. Fork репозитория на GitHub
# 2. Клонировать ваш fork
git clone https://github.com/your-username/nextstat.git
cd nextstat

# 3. Добавить upstream remote
git remote add upstream https://github.com/nextstat/nextstat.git

# 4. Установить зависимости
cargo build --workspace
cargo test --workspace

# 5. Установить pre-commit hooks (опционально)
# Будет добавлено в Phase 0
```

## Процесс разработки

### 1. Создать ветку

```bash
git checkout -b feature/your-feature-name
```

**Naming convention:**
- `feature/` — новая функциональность
- `bugfix/` — исправление бага
- `docs/` — изменения в документации
- `refactor/` — рефакторинг без изменения API

### 2. Следовать TDD (Test-Driven Development)

**Обязательно для всех изменений в коде:**

1. **Написать failing test**
   ```bash
   # Добавить тест в соответствующий файл
   cargo test --package ns-core --test your_test -- --nocapture
   # Должен FAIL
   ```

2. **Реализовать минимальный код**
   ```rust
   // Написать минимальную реализацию
   ```

3. **Запустить тест снова**
   ```bash
   cargo test --package ns-core --test your_test
   # Должен PASS
   ```

4. **Refactor (если нужно)**
   ```bash
   cargo test --workspace  # Все тесты должны проходить
   ```

5. **Commit с DCO sign-off**
   ```bash
   git add .
   git commit -s -m "feat(ns-core): add new functionality"
   ```

### 3. Coding Standards

#### Rust

- **Style:** Используйте `cargo fmt` перед коммитом
- **Linting:** Исправьте все предупреждения `cargo clippy`
- **Documentation:** Все public API должны иметь docstrings
- **Tests:** Покрытие ≥ 80% для новых модулей
- **Error handling:** Используйте `Result<T, Error>`, избегайте `panic!`

```rust
/// Compute negative log-likelihood
///
/// # Arguments
///
/// * `params` - Parameter values
///
/// # Returns
///
/// Negative log-likelihood value
///
/// # Errors
///
/// Returns error if computation fails
pub fn nll(&self, params: &[f64]) -> Result<f64> {
    // Implementation
}
```

#### Python

- **Style:** PEP 8, используйте `black` для форматирования
- **Type hints:** Обязательны для всех функций
- **Docstrings:** Google style

```python
def fit(self, initial_params: list[float]) -> FitResult:
    """Perform maximum likelihood fit.

    Args:
        initial_params: Initial parameter values.

    Returns:
        Fit result with best-fit parameters and uncertainties.

    Raises:
        ValueError: If initial_params is empty.
    """
```

### 4. Commit Messages

Следуйте [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

[optional body]

[optional footer]

Signed-off-by: Your Name <your.email@example.com>
```

**Types:**
- `feat` — новая функциональность
- `fix` — исправление бага
- `docs` — только документация
- `test` — добавление тестов
- `refactor` — рефакторинг без изменения API
- `perf` — оптимизация производительности
- `chore` — maintenance задачи

**Scopes:** `ns-core`, `ns-compute`, `ns-inference`, `ns-translate`, `ns-viz`, `ns-cli`, `ns-py`

**Примеры:**
```
feat(ns-inference): implement L-BFGS optimizer
fix(ns-compute): correct gradient calculation for Poisson
docs(README): update installation instructions
test(ns-core): add tests for error handling
```

## DCO Sign-off

**ОБЯЗАТЕЛЬНО:** Все коммиты должны быть подписаны DCO (Developer Certificate of Origin).

### Что такое DCO?

DCO — это легковесная альтернатива CLA (Contributor License Agreement). Подписывая коммит, вы подтверждаете, что имеете право вносить этот код под лицензией проекта (AGPL-3.0).

Полный текст: [DCO.md](DCO.md)

### Как подписать коммит

**Автоматически (рекомендуется):**
```bash
git commit -s -m "your commit message"
```

**Вручную:**
```bash
git commit -m "your commit message

Signed-off-by: Your Name <your.email@example.com>"
```

### Проверка sign-off

```bash
git log --show-signature
```

Каждый коммит должен содержать строку:
```
Signed-off-by: Your Name <your.email@example.com>
```

### Если забыли подписать

**Последний коммит:**
```bash
git commit --amend --signoff
```

**Несколько коммитов:**
```bash
git rebase --signoff HEAD~3  # последние 3 коммита
git push --force-with-lease origin your-branch
```

## Процесс Pull Request

### 1. Проверить перед созданием PR

- [ ] Все тесты проходят: `cargo test --workspace`
- [ ] Нет clippy warnings: `cargo clippy --workspace -- -D warnings`
- [ ] Код отформатирован: `cargo fmt --check`
- [ ] Все коммиты подписаны DCO
- [ ] Документация обновлена (если нужно)
- [ ] Добавлены тесты для новой функциональности

### 2. Создать Pull Request

1. Push в ваш fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Открыть PR на GitHub: `base: main` ← `compare: your-branch`

3. Заполнить шаблон PR:
   ```markdown
   ## Описание
   [Краткое описание изменений]

   ## Тип изменений
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Чеклист
   - [ ] Тесты проходят
   - [ ] Код отформатирован (cargo fmt)
   - [ ] Нет clippy warnings
   - [ ] DCO sign-off на всех коммитах
   - [ ] Документация обновлена
   - [ ] Следовал TDD процессу

   ## Связанные Issues
   Closes #123
   ```

### 3. Code Review

- Maintainers проверят ваш код и оставят комментарии
- Внесите запрошенные изменения
- Push изменений автоматически обновит PR

### 4. Merge

После одобрения maintainer'ом ваш PR будет влит в `main`.

## Тестирование

### Типы тестов

1. **Unit tests** — тесты отдельных функций/модулей
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_nll_calculation() {
           let backend = CpuBackend::new();
           let result = backend.nll(&[1.0, 2.0]);
           assert!(result.is_ok());
       }
   }
   ```

2. **Integration tests** — тесты взаимодействия между модулями
   ```rust
   // tests/integration_test.rs
   use ns_inference::MaximumLikelihoodEstimator;
   use ns_compute::CpuBackend;

   #[test]
   fn test_mle_with_cpu_backend() {
       // Test full workflow
   }
   ```

3. **Doc tests** — примеры в документации
   ```rust
   /// ```
   /// use ns_core::ComputeBackend;
   /// let backend = CpuBackend::new();
   /// assert_eq!(backend.name(), "CPU");
   /// ```
   ```

### Запуск тестов

```bash
# Все тесты
cargo test --workspace

# Конкретный package
cargo test --package ns-core

# Конкретный тест
cargo test --package ns-core test_name

# С выводом
cargo test --package ns-core -- --nocapture

# Только doc tests
cargo test --doc
```

### Требования к покрытию

- Новые модули: ≥ 80% покрытие
- Критические компоненты (ns-core, ns-compute): ≥ 90%
- Bug fixes: добавить regression test

## Документация

### Типы документации

1. **Code documentation** (обязательно для public API)
   ```rust
   /// Brief description.
   ///
   /// Detailed description with examples.
   ///
   /// # Arguments
   ///
   /// * `param` - Description
   ///
   /// # Returns
   ///
   /// Description of return value
   ///
   /// # Errors
   ///
   /// When this function returns error
   ///
   /// # Examples
   ///
   /// ```
   /// let result = function(param);
   /// ```
   pub fn function(param: Type) -> Result<Output> {
       // Implementation
   }
   ```

2. **User documentation** (для новых возможностей)
   - Обновить README.md
   - Добавить примеры в docs/
   - Обновить CHANGELOG.md (maintainers сделают)

3. **Architecture documentation** (для больших изменений)
   - Создать RFC в docs/rfcs/
   - Обновить docs/architecture/

## Вопросы?

- Создайте Issue с меткой `question`
- Email: dev@nextstat.io
- Документация: https://docs.nextstat.io

---

**Спасибо за ваш вклад в NextStat!** 🚀
