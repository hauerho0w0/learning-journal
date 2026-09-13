---
layout: default
title: 目錄
---

# 高中學習歷程

紀錄高中三年的學習，內容涵蓋所有有興趣的人事物。

{%- assign docs = site.pages | where_exp: "p", "p.date" | sort: "date" | reverse -%}
{%- assign groups = docs | group_by: "date" -%}
{%- for g in groups %}

## {{ g.name }}

{% assign items = g.items | sort: "order" -%}
{%- for d in items %}
- [{{ d.title }}]({{ d.url | relative_url }}){% if d.summary %} — {{ d.summary }}{% endif %}
{%- endfor %}
{%- endfor %}
