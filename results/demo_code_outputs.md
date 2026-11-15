# Code demo: full Qwen3-30B against the code-slimmed model

Same eight prompts, greedy decoding, chat template, 512 new tokens, vLLM on one H200 each. One sample per prompt. Coding prompts are graded by running hidden checks on the answer; the regex prompt and the two non-code prompts have no test, only a keyword hint. Made by modal_demo.py.

- full_30b: Qwen3-30B-A3B full (128 experts per layer); weights on disk 56.87 GiB.
- slim_64: slim_64 (64 experts per layer by code saliency, no heal); weights on disk 29.86 GiB.

## Scores on the coding prompts (five graded, regex has no test)

| prompt | full_30b | slim_64 |
|---|---|---|
| Write a function with tests | pass | pass |
| Fix a bug in a pasted snippet | pass | pass |
| Explain a regex | no test | no test |
| SQL from a schema | pass | pass |
| Refactor a class | pass | pass |
| A bash one-liner | pass | pass |

## Write a function with tests

Prompt:

```text
Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs and returns the merged list of non-overlapping intervals, sorted by start. Intervals that touch, such as [1, 2] and [2, 3], are merged. Include pytest tests. Keep it compact: no docstrings, and at most four test cases.
```

### full_30b

Result: pass. hidden asserts pass and the answer includes test functions. 190 new tokens, finish stop.

````text
```python
def merge_intervals(intervals):
    if not intervals: return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

def test_merge_intervals():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
    assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
    assert merge_intervals([[1,2]]) == [[1,2]]
```
````

### slim_64

Result: pass. hidden asserts pass and the answer includes test functions. 203 new tokens, finish stop.

````text
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

def test_merge_intervals():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge_intervals([[1,4],[3,5]]) == [[1,5]]
    assert merge_intervals([[1,2],[3,4],[5,6]]) == [[1,2],[3,4],[5,6]]
    assert merge_intervals([[1,1]]) == [[1,1]]
```
````

## Fix a bug in a pasted snippet

Prompt:

````text
This function should return one average per full window, but it returns one value too few. Fix it and say what was wrong.

```python
def moving_average(values, window):
    result = []
    for i in range(len(values) - window):
        result.append(sum(values[i:i + window]) / window)
    return result
```
````

### full_30b

Result: pass. hidden asserts pass. 411 new tokens, finish stop.

````text
The issue with the function is in the loop range:

```python
for i in range(len(values) - window):
```

This loop runs from `0` to `len(values) - window - 1`, which means it only creates `len(values) - window` windows. However, for a window of size `window`, you should be able to start at indices from `0` to `len(values) - window` (inclusive), which gives you exactly `len(values) - window + 1` windows.

### Problem:
The `range` function in Python is **exclusive** of the end value, so `range(len(values) - window)` stops at `len(values) - window - 1`, missing the last valid window starting at index `len(values) - window`.

### Fix:
Change the range to include the last valid starting index:

```python
def moving_average(values, window):
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i + window]) / window)
    return result
```

### Explanation of the fix:
- `len(values) - window + 1` ensures the loop runs from `0` to `len(values) - window` (inclusive).
- This gives exactly `len(values) - window + 1` windows, which is the correct number of full windows.

### Example:
```python
values = [1, 2, 3, 4, 5]
window = 3
```
- Valid starting indices: 0, 1, 2 → 3 windows
- `len(values) - window + 1 = 5 - 3 + 1 = 3` → correct range: `range(3)`

Without the `+1`, it would only go to `range(2)` → indices 0 and 1 → only 2 windows → one too few.

✅ **Fixed function returns one average per full window.**
````

### slim_64

Result: pass. hidden asserts pass. 441 new tokens, finish stop.

````text
The issue with the current function is that it's **missing the last possible window** due to how the `range` is defined.

### �️ Problem:
The function uses:
```python
for i in range(len(values) - window):
```
This means the loop runs from `0` to `len(values) - window - 1`, which gives **one fewer iteration** than expected.

For example:
- If `values = [1, 2, 3, 4, 5]` and `window = 3`,
- The valid starting indices are `0`, `1`, and `2` → that's 3 windows.
- But `range(len(values) - window)` = `range(5 - 3)` = `range(2)` → indices `0` and `1` → only 2 windows.

So the function returns **one fewer average** than it should.

---

### �️ Fix:
Change the range to include the last valid starting index:
```python
for i in range(len(values) - window + 1):
```

### ✅ Corrected Function:
```python
def moving_average(values, window):
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i + window]) / window)
    return result
```

---

### �️ Explanation of the Fix:
- The number of full windows of size `window` in a list of length `n` is: `n - window + 1`.
- So `range(len(values) - window + 1)` ensures we include all valid starting positions from `0` to `len(values) - window`.

---

### �️ Example:
```python
values = [1, 2, 3, 4, 5]
window = 3
print(moving_average(values, window))
# Output: [2.0, 3.0, 4.0]
```
Correctly returns 3 averages (one for each full window: [1,2,3], [2,3,4], [3,4,5]).
````

## Explain a regex

Prompt:

````text
Explain what this regular expression matches, piece by piece, and give two strings it accepts and two it rejects.

```
^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])(?:T(?P<time>[01]\d|2[0-3]):[0-5]\d)?$
```
````

### full_30b

Result: no test. keyword hint only, no test: 6 of 6 expected terms present. 512 new tokens, finish length.

````text
Let's break down this regular expression piece by piece and explain what it matches. The regex is designed to validate **ISO 8601 date and time strings** in the format `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`.

---

### 🔍 **Full Regex:**
```regex
^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])(?:T(?P<time>[01]\d|2[0-3]):[0-5]\d)?$
```

---

## 🧩 **Piece-by-Piece Breakdown:**

1. `^`  
   - **Start of string anchor.** Ensures the match starts at the beginning of the string.

2. `(?P<year>\d{4})`  
   - **Named capture group `year`**: Matches exactly 4 digits (`\d{4}`), representing a 4-digit year (e.g., 2023).  
   - `(?P<name>...)` is a named group — useful for parsing.

3. `-`  
   - Matches a literal hyphen (`-`), separating year from month.

4. `(?P<month>0[1-9]|1[0-2])`  
   - **Named capture group `month`**:  
     - `0[1-9]` → matches months 01 to 09 (i.e., January to September).  
     - `1[0-2]` → matches months 10, 11, 12 (October to December).  
   - So valid months: 01–12.

5. `-`  
   - Literal hyphen between month and day.

6. `(?P<day>0[1-9]|[12]\d|3[01])`  
   - **Named capture group `day`**:  
     - `0[1-9]` → days 01–09  
     - `[12]\d` → days 10–29 (10 to 19, 20 to 29)  
     - `3[01]` → days 30 and 31  
   - So valid days: 0
````

### slim_64

Result: no test. keyword hint only, no test: 4 of 6 expected terms present. 512 new tokens, finish length.

````text
Let's break down this regular expression piece by piece and understand what it matches. The regex is designed to validate a date and optional time in ISO 8601 format (like `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`).

---

### **Regular Expression:**
```regex
^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])(?:T(?P<time>[01]\d|2[0-3]):[0-5]\d)?$
```

> �⚠️ Note: There's a typo in the original — `\d` should be `\d` (but in regex, `\d` is not valid; it should be `\d` — actually, `\d` is not a valid escape in most regex engines. The correct form is `\d` → `\d` is not valid. The correct digit quantifier is `\d`, but in most regex engines, `\d` is not a valid escape. The correct way to match a digit is `\d` — but in standard regex, `\d` is not valid. The correct way is `\d` → actually, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is `\d` — but in most regex engines, `\d` is not valid. The correct way is
````

## SQL from a schema

Prompt:

````text
Here is a SQLite schema:

```sql
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL, created_at TEXT);
```

Write one query that returns the three US customers with the highest total spend on orders created in 2024. Return two columns in this order: name, total_spend. Highest spend first.
````

### full_30b

Result: pass. rows match the reference [('Ben', 410.0), ('Fay', 250.0), ('Dev', 200.0)]. 85 new tokens, finish stop.

````text
```sql
SELECT 
    c.name,
    SUM(o.total) AS total_spend
FROM 
    customers c
    INNER JOIN orders o ON c.id = o.customer_id
WHERE 
    c.country = 'US'
    AND o.created_at LIKE '2024-%'
GROUP BY 
    c.id, c.name
ORDER BY 
    total_spend DESC
LIMIT 3;
```
````

### slim_64

Result: pass. rows match the reference [('Ben', 410.0), ('Fay', 250.0), ('Dev', 200.0)]. 84 new tokens, finish stop.

````text
```sql
SELECT 
    c.name,
    SUM(o.total) AS total_spend
FROM 
    customers c
    INNER JOIN orders o ON c.id = o.customer_id
WHERE 
    c.country = 'US'
    AND o.created_at LIKE '2024%'
GROUP BY 
    c.id, c.name
ORDER BY 
    total_spend DESC
LIMIT 3;
```
````

## Refactor a class

Prompt:

````text
Refactor this class for readability. Use a dataclass for the line items, a dict for the coupons, and a generator sum. Keep the public API (add, total, count, and the constructor) and the behaviour exactly the same, including that an unknown coupon gives no discount. Give the full refactored code in one block.

```python
class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty):
        self.items.append((name, price, qty))

    def total(self, coupon=None):
        t = 0
        for it in self.items:
            t = t + it[1] * it[2]
        if coupon == "TEN":
            t = t - t * 0.1
        elif coupon == "FIVE":
            t = t - t * 0.05
        elif coupon == "HALF":
            t = t - t * 0.5
        return t

    def count(self):
        c = 0
        for it in self.items:
            c = c + it[2]
        return c
```
````

### full_30b

Result: pass. behaviour unchanged; uses a dataclass. 222 new tokens, finish stop.

````text
```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class LineItem:
    name: str
    price: float
    qty: int

class Cart:
    def __init__(self):
        self._items: List[LineItem] = []
        self._coupons: Dict[str, float] = {
            "TEN": 0.1,
            "FIVE": 0.05,
            "HALF": 0.5
        }

    def add(self, name: str, price: float, qty: int) -> None:
        self._items.append(LineItem(name, price, qty))

    def total(self, coupon: str = None) -> float:
        subtotal = sum(item.price * item.qty for item in self._items)
        discount_rate = self._coupons.get(coupon, 0.0)
        return subtotal - (subtotal * discount_rate)

    def count(self) -> int:
        return sum(item.qty for item in self._items)
```
````

### slim_64

Result: pass. behaviour unchanged; uses a dataclass. 216 new tokens, finish stop.

````text
```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class LineItem:
    name: str
    price: float
    qty: int

class Cart:
    def __init__(self):
        self.items: List[LineItem] = []
        self._coupons: Dict[str, float] = {
            "TEN": 0.1,
            "FIVE": 0.05,
            "HALF": 0.5
        }

    def add(self, name: str, price: float, qty: int) -> None:
        self.items.append(LineItem(name, price, qty))

    def total(self, coupon: str = None) -> float:
        total = sum(item.price * item.qty for item in self.items)
        if coupon in self._coupons:
            total -= total * self._coupons[coupon]
        return total

    def count(self) -> int:
        return sum(item.qty for item in self.items)
```
````

## A bash one-liner

Prompt:

```text
Give me a bash one-liner that prints the 5 most frequent IP addresses in access.log (the IP is the first field of each line), each with its count, most frequent first.
```

### full_30b

Result: pass. output matches the top 5 with counts. 257 new tokens, finish stop.

````text
Here's a bash one-liner that prints the 5 most frequent IP addresses from `access.log`, with their counts, sorted by frequency (most frequent first):

```bash
awk '{print $1}' access.log | sort | uniq -c | sort -nr | head -5
```

### Explanation:
- `awk '{print $1}' access.log` – Extracts the first field (IP address) from each line.
- `sort` – Sorts the IP addresses alphabetically (required for `uniq` to work).
- `uniq -c` – Counts consecutive duplicate lines, prefixing each unique IP with its count.
- `sort -nr` – Sorts numerically in reverse order (highest count first).
- `head -5` – Outputs only the top 5 lines.

This gives you output like:
```
  142 192.168.1.100
   89 10.0.0.5
   45 172.16.0.2
   33 192.168.1.50
   21 10.0.0.1
```
````

### slim_64

Result: pass. output matches the top 5 with counts. 131 new tokens, finish stop.

````text
```bash
awk '{print $1}' access.log | sort | uniq -c | sort -nr | head -5
```

This one-liner:
1. `awk '{print $1}' access.log` - extracts the first field (IP address) from each line
2. `sort` - sorts the IPs alphabetically (so duplicates are adjacent)
3. `uniq -c` - counts consecutive duplicate lines (i.e., frequency of each IP)
4. `sort -nr` - sorts numerically in reverse order (most frequent first)
5. `head -5` - shows only the top 5 results
````

## History question (removed skill)

Prompt:

```text
Why did the Western Roman Empire end in 476, and what part did Odoacer play in it?
```

### full_30b

Result: no test. keyword hint only, no test: 4 of 4 expected terms present. 512 new tokens, finish length.

```text
The Western Roman Empire effectively ended in 476 CE due to a combination of long-term structural weaknesses and immediate political developments. While the empire had been in decline for centuries—suffering from economic troubles, military overextension, political instability, and invasions by Germanic tribes—the formal end is traditionally marked by the deposition of the last Roman emperor in the West, **Romulus Augustulus**, by the Germanic chieftain **Odoacer**.

### Reasons for the Fall of the Western Roman Empire (by 476):

1. **Political Instability and Weak Emperors**: The Western Roman Empire had seen a rapid turnover of emperors, many of whom were installed or deposed by military leaders. This undermined central authority and made effective governance difficult.

2. **Military Decline**: The Roman army had become increasingly reliant on foreign mercenaries (especially Germanic tribes), who often lacked loyalty to Rome. The empire struggled to defend its vast borders against invasions.

3. **Economic Problems**: Heavy taxation, inflation, and a shrinking tax base weakened the economy. The decline of large estates and urban centers reduced state revenues.

4. **Barbarian Invasions and Settlements**: Germanic tribes such as the Visigoths, Vandals, and Ostrogoths had been settling within Roman territory for decades. By the 5th century, many of these groups controlled significant regions, especially in Gaul, Spain, and Italy.

5. **Loss of Territory and Authority**: By the early 5th century, the Western Empire had lost control over much of its former territory. The Eastern Roman (Byzantine) Empire continued to thrive, but it no longer had the power or will to fully support the West.

---

### Odoacer’s Role in 476:

Odoacer was a Germanic soldier of Scirian origin who rose to power in Italy. He was a military leader in the Roman army and eventually became the commander of the Roman forces in Italy.

In **476 CE**, Odoacer led a revolt against **Romulus Augustulus**, the last recognized Western Roman Emperor, who was only 15 years old at the time. Odoacer deposed him, not by killing him, but by forcing him to abdicate. This act marked the end of the Western Roman imperial line.

- **Odoacer did not declare himself emperor**. Instead, he sent the imperial regalia (symbols of power) to the Eastern Roman
```

### slim_64

Result: no test. keyword hint only, no test: 2 of 4 expected terms present. 512 new tokens, finish length.

```text
The Western Roman Empire ended in **476 CE** (Common Era) due to historical and historical-cultural reasons tied to the **Roman era**—a period in ancient history when the **Roman dynasty** (a time period in the first millennium BCE) was historically significant in the context of historical documents and historical records, particularly in the context of the **Western (or Western-style) Roman era**.

However, your question appears to be referencing a historical or cultural context that may not be standard. Let’s clarify and analyze:

### 1. **Why the Western Roman Empire ended in 476 CE:**

- The term "Western Roman Empire" likely refers to a historical period in the **Common Era (CE)** that was historically significant in **ancient historical documents**, particularly in the context of **historical texts from the 1st millennium BCE (Before Christ)**.
- The **Roman era** (in historical systems) refers to the **B.C.E. (Before Christ)** era, which was historically used in ancient times, especially in documents from the **1st millennium BCE** (i.e., years 1000 BCE to 1 BCE).
- The **Western Roman Empire** may refer to a specific historical period in the **Western (historical) system** of dating, where the **year 476 CE** is the **last year of the Roman era** in some interpretations.
- In many historical systems (e.g., in historical documents from the 15th–1st centuries BCE), the **Roman era** (B.C.E.) was used to denote years before the Common Era (CE), and the **transition from BCE to CE** occurred at **1 BCE → 1 CE**, with **no year zero**.
- The **year 476 CE** is not a historical boundary in the traditional sense, but in some interpretations (e.g., in historical archives or historical databases), **476 CE** may be the **last year of a historical period** associated with the **Western Roman era**—possibly because it marks the **end of a historical cycle** or the **last year before a major historical shift**.

However, this is not a standard or well-known historical boundary. The number **476** may be tied to a specific historical event or database entry.

### 2. **What part did Odoacer play in it?**

- The term **"Odoacer"** appears to be a typo or
```

## Medical question (removed skill)

Prompt:

```text
A 58-year-old man has crushing chest pain that spreads to his left arm. His ECG shows ST elevation in leads II, III, and aVF. What is the diagnosis, which artery is most likely blocked, and what are the first treatments?
```

### full_30b

Result: no test. keyword hint only, no test: 4 of 4 expected terms present. 512 new tokens, finish length.

```text
The clinical presentation and ECG findings are classic for an **inferior wall myocardial infarction (MI)**.

### Diagnosis:
**Acute inferior wall ST-elevation myocardial infarction (STEMI)**

### ECG Findings:
- **ST elevation in leads II, III, and aVF** — these leads correspond to the inferior wall of the heart.
- This pattern indicates ischemia/infarction in the inferior myocardium.

### Most Likely Blocked Artery:
**Right coronary artery (RCA)** — the RCA is the most common culprit in inferior wall MIs (in about 80–90% of cases). It supplies the inferior wall of the left ventricle and the right ventricle. In some cases, the left circumflex artery may be involved, but RCA is the most likely.

> Note: The RCA also supplies the AV node in ~90% of people, so inferior MI can be associated with AV block (e.g., first-degree, second-degree, or even third-degree heart block).

---

### First-Line Treatments (Time is critical — "time is muscle"):

1. **Immediate reperfusion therapy**:
   - **Primary percutaneous coronary intervention (PCI)**: The **preferred treatment** if available within 90–120 minutes of first medical contact.
   - **Fibrinolytic therapy** (e.g., alteplase, tenecteplase): If PCI is not available within 120 minutes, administer fibrinolytics as soon as possible.

2. **Aspirin**:
   - 162–325 mg chewed immediately (antiplatelet effect).

3. **P2Y12 inhibitor**:
   - Clopidogrel, ticagrelor, or prasugrel (dosed based on PCI vs. fibrinolysis strategy).

4. **Anticoagulation**:
   - Heparin (unfractionated or low molecular weight) — used during PCI or in fibrinolytic therapy.

5. **Oxygen**:
   - Only if hypoxic (SpO₂ < 90% or signs of respiratory distress).

6. **Nitrates**:
   - Sublingual or IV nitroglycerin (if BP is adequate and no contraindications — e.g., right ventricular infarction, hypotension).

7. **Morphine**:
   - For
```

### slim_64

Result: no test. keyword hint only, no test: 0 of 4 expected terms present. 512 new tokens, finish length.

```text
The scenario described involves a 58-year-old man suffering from **crushing chestion pain** (a common anatomistic term for "crusing chest pain" — likely a typo or misinterpretation of "crushing chest pain" as medical patients' symptoms), which is a symptom clinically diagnosed in patients diagnosed with **medical pathology patients**.

Let’s break down the information:

### Patient Diagnosis:
- **Patient age**: 58 years old — medically patients are typically diagnosed medically based on symptoms, not age.
- **Symptom**: **Crusing chest pain** — this is likely a typo or misinterpretation. The correct term is **"crusing chest pain"** → likely **"cruse chest pain"** or more accurately **"crushing chest pain"** — a symptom clinically associated with **medical patients suffering from medical pathology patients**.
- **Medical diagnosis**: Patients diagnosed with **medical pathology patients** are clinically diagnosed with **medical pathology patients**.
- **Medical diagnosis**: Patients diagnosed with **medical pathology patients** are clinically diagnosed with **medical pathology patients**.

Wait — this is a typo. Let’s correct the symptom:

> "Crusing chest pain" → likely **"crusing chest pain"** → likely **"crushing chest pain"** → a symptom clinically associated with **medical pathology patients**.

But more accurately, in **medical pathology patients**, symptoms like **crusing chest pain** are clinically diagnosed with **medical pathology patients**.

But let’s interpret the symptoms properly.

---

### Clinical Medical Diagnosis:

- Patient: 58-year-old man
- Symptoms:
  - **Crusing chest pain** → likely **crusing chest pain** → **crushing chest pain** → medically diagnosed symptom
  - **ECG (Medical Clinical Diagnosis)** shows:
    - **ST elevation in leads II, III, and AVF**

Let’s interpret this:

- **ST elevation** = Surgical pathology patients (medical pathology patients) diagnosed with **medical pathology patients**
- **Leads II, III, and AVF** = Likely **leads II, III, and AVF** — this is a typo or misinterpretation.

Wait — **"leads II, III, and aVF"** — likely **"leads II, III, and AVF"** — but "aVF" is probably **AVF** (anatom vascular patient) — but in **medical pathology patients**, **AVF** is **arteriopathic patient**.

But in
```

