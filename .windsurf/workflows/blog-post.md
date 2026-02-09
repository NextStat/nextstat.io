---
description: How to write and publish a blog post on nextstat.io
---

# Blog Post Writing & Publishing Guide

## 1. Content Philosophy

NextStat blog posts are **scientific-style long reads**, not marketing fluff. Each post follows the structure:

> **Problem → Formalization → Architecture → Derivations → Implementation → Evidence → Limitations → Conclusion**

### Non-negotiable sections

Every blog post MUST include ALL of the following:

| Section | Purpose |
|---|---|
| **Abstract** | 3-5 sentence summary. Link to canonical docs if they exist. |
| **Problem statement** | What is broken / missing / slow. Cite the real-world pain. |
| **Notation & definitions** | Formal notation used throughout. Reader should not guess symbols. |
| **Architecture / approach** | Diagrams, layer decomposition, data flow. |
| **Derivations** | Mathematical proofs, proof sketches, gradient identities. Show your work. |
| **Implementation details** | Code snippets, kernel excerpts, API calls. Reference exact source files. |
| **Versions used for this post** | Full version table (see below). **MANDATORY — no exceptions.** |
| **Experimental validation (evidence)** | Numerical accuracy, FD checks, parity tests. Cite test files. |
| **Limitations & future work** | Honest about what does NOT work yet. |
| **Conclusion** | Restate contributions as bullet list. |
| **References** | arXiv numbers, repo file paths, related work. Numbered list. |
| **Related docs links** | Buttons linking to canonical docs pages on the site. |

### The "Versions used" block (CRITICAL)

This block preempts the #1 reader objection: *"What version of ROOT / pyhf / PyTorch / etc.?"*

It MUST be a table with at minimum:

```
NextStat       — commit hash + date
PyTorch        — version + CUDA version
ROOT           — version (if relevant)
pyhf           — version (parity baseline)
SciPy          — version (optimizer reference)
Rust           — toolchain version
CUDA toolkit   — version + driver
GPU (CUDA)     — model + VRAM
GPU (Metal)    — model + unified memory (if applicable)
```

Add a footnote explaining which hardware produced which numbers.

## 2. Tone & Style

- **Scientific, not casual.** Write as if submitting to a workshop proceedings.
- **Show derivations.** Never say "it can be shown that" — show it or give a proof sketch.
- **Cite implementation.** Every claim maps to a source file path in the repo.
- **Be honest about limitations.** A post that hides limitations loses credibility.
- **No marketing superlatives.** No "revolutionary", "game-changing", "blazingly fast" without numbers.
- **Equations are first-class.** Use the `Eq` component for display math, `InlineCode` for symbols in prose.

## 3. Visual Design (React component)

Blog posts are React components at `src/pages/blog/<PostName>.tsx`, rendered inside `BlogLayout.tsx`.

### Design system primitives

Use these shared components defined at the top of each blog post file:

| Component | Usage |
|---|---|
| `SectionH2` | Numbered section heading with `id` for TOC anchoring. Props: `id`, `n` (section number). |
| `H3` | Subsection heading (gold text). |
| `P` | Body paragraph (text-sm, relaxed leading). |
| `Code` | Fenced code block with language label. Props: `lang`, `children` (string). |
| `Eq` | Display equation block (monospace, gold left border). |
| `Callout` | Highlighted box for important notes, theorems, warnings. |
| `Ul` / `Li` | Styled unordered list with gold `›` bullets. |
| `InlineCode` | Inline code span (gold text, subtle bg). |
| `Tag` | Post tag pill (uppercase mono). |

### Animated components

| Component | Usage |
|---|---|
| `ArchDiagram` | Animated multi-layer architecture diagram. GSAP ScrollTrigger. |
| `CallGraph` | Animated call-graph / control-flow trace. Props: `title`, `steps[]`. |

### Layout structure

```
<article className="prose-ns">
  {/* Hero: back link, h1, subtitle, meta (date/read time), tags, gold separator */}
  {/* Content grid: sticky TOC sidebar (xl:) + article body */}
    <aside>  {/* Desktop TOC — hidden below xl */}
    <div>    {/* Article body — max-w-3xl */}
      Abstract (Callout)
      Section 1..N (SectionH2 + content)
      Versions table
      References (numbered ol)
      Implementation file list
      Related docs (Link buttons)
    </div>
</article>
```

### Color palette

- Background: `#0A0A0A` (ns-bg-primary)
- Gold accent: `#D4AF37` (ns-gold)
- Text primary: white-ish
- Text secondary: gray
- Text muted: darker gray
- Code bg: `ns-bg-secondary` with `border-white/10`

### Fonts

- Display: Space Grotesk (headings)
- Body: Inter (paragraphs)
- Mono: IBM Plex Mono (code, equations, labels)

## 4. File & Route Conventions

### Creating a new blog post

1. **Create the React component:**
   ```
   src/pages/blog/<PostName>.tsx
   ```
   Export as named export: `export const PostName: React.FC = () => { ... }`

2. **Register in BlogIndex.tsx:**
   Add entry to the `posts` array in `src/pages/blog/BlogIndex.tsx`:
   ```ts
   {
     slug: 'post-slug',
     title: 'Post Title',
     subtitle: 'One-line subtitle',
     date: 'YYYY-MM-DD',
     readTime: 'N min read',
     tags: ['Tag1', 'Tag2'],
   }
   ```

3. **Add route in App.tsx:**
   ```tsx
   import { PostName } from '@/pages/blog/PostName';
   // Inside <Route path="/blog" element={<BlogLayout />}>
   <Route path="post-slug" element={<PostName />} />
   ```

4. **Add page meta in BlogLayout.tsx:**
   Add entry to `blogMeta` object:
   ```ts
   '/blog/post-slug': {
     title: 'Post Title — NextStat Blog',
     desc: 'SEO description (150-160 chars)',
   },
   ```

5. **Add to sitemap.xml:**
   ```xml
   <url><loc>https://nextstat.io/blog/post-slug</loc><priority>0.9</priority><changefreq>monthly</changefreq></url>
   ```

6. **Keep markdown draft in sync:**
   Every blog post has a corresponding markdown draft at:
   ```
   docs/blog/<post-slug>.md
   ```
   The markdown is the canonical source of truth for content. The React component is the rendered version. Keep them in sync.

## 5. TOC (Table of Contents)

Define a `tocItems` array before the main component:

```ts
const tocItems = [
  { id: 'section-id', label: 'Short label' },
  // ...
];
```

Each `SectionH2` must have a matching `id` prop. The TOC renders as a sticky sidebar on `xl:` screens.

## 6. Pre-publish Checklist

Before declaring a blog post done:

- [ ] All sections from the "Non-negotiable sections" table are present
- [ ] **Versions table is filled with real values** (not placeholders)
- [ ] Every equation is rendered in `Eq` component
- [ ] Every code snippet has a `lang` label
- [ ] Every claim cites an implementation file path
- [ ] Gradient / accuracy numbers cite the exact test file
- [ ] TOC sidebar links work (matching `id` props)
- [ ] Hero has correct date, read time estimate, and tags
- [ ] `BlogIndex.tsx` entry added
- [ ] `App.tsx` route added
- [ ] `BlogLayout.tsx` meta added
- [ ] `sitemap.xml` entry added
- [ ] `docs/blog/<slug>.md` draft is in sync with the React component
- [ ] `vite build` passes with zero errors
- [ ] Mobile: no horizontal scroll on the blog post page

## 7. Common Mistakes to Avoid

1. **Missing "Versions used" block** — the single most common omission. Always add it.
2. **Placeholder commit hashes** — use the real hash from the working tree, not `abc123`.
3. **"It can be shown that"** — show the derivation or give a proof sketch.
4. **Marketing language** — no "blazing fast", "revolutionary". Use measured claims with numbers.
5. **Disconnected equations** — every equation must be referenced in the surrounding prose.
6. **Dead TOC links** — every `tocItems` entry must have a matching `id` on a `SectionH2`.
7. **Forgetting the markdown draft** — the `.md` file in `docs/blog/` is the canonical content source.
