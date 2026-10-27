# Labelling instructions: token name and symbol strings

**Protocol version 2.2.** Once the first rater has started, this document is not edited. Any change
is issued as a new version. (2.1 permits machine translation of the two texts; see rule 1. 2.2 adds
the web page as a way to answer. The rows and the questions are unchanged.)

## What you receive

You get a link to a web page that shows about 400 rows one at a time and asks the questions below.
Each row is one token contract on a public blockchain, shown as the two short texts the contract
reports about itself: its **name** and its **symbol**, exactly as written. Nothing else about the
contract is shown.

The page keeps your answers in your browser as you go, so you can stop and come back on the same
computer and browser. If you prefer a spreadsheet, ask for `sample_blinded.csv` instead. It holds
the same rows, is UTF-8, and opens in Excel, LibreOffice or Google Sheets.

This is part of a measurement study. Your labels will be compared with an automated method that you
will not see.

## Rules

1. **Judge the two texts only.** Do not look up the contract, the domain or the project. **Do not
   open any link.** Some of them may be malicious.
   - **Translation is allowed.** If a text is in a language you cannot read, you may paste **that
     text** into a machine translator (for example Google Translate or DeepL) to understand it.
     Translate only the name and symbol, and never follow a link the translator shows.
   - When you used a translator for a row, tick **I used a translator** on the web page, or write
     `translated` in the row's `notes` cell in the spreadsheet.
2. **Work alone.** Do not discuss any row with anyone else labelling this sheet until everyone has
   handed in.
3. **Fill in every row.** Use `unclear` whenever you cannot decide. Do not guess.
4. **Read lookalike characters as the letters they imitate.** For example, a Cyrillic "с" counts
   as a Latin "c". Emoji, tick marks and decorative symbols mean nothing by themselves.
5. Expect it to take roughly 45–60 minutes.

## What to fill in

On the web page you click the answers. In the spreadsheet, type exactly the values listed. Each
column takes one value per row.

| Column | Question | Values |
|---|---|---|
| `destination` | Do the texts tell the reader **where to go**? That means a web address or domain name, a link, a messaging handle (`t.me/…`, `@channel`), a shortened link, or a disguised form of any of these (`example[.]com`, `example dot com`, `hxxp://…`). A domain counts even if it looks like a project's own name. | `yes` · `no` · `unclear` |
| `call_to_action` | Do the texts **urge the reader to do something, or offer something in return**? That means an instruction (visit, claim, connect, redeem…), a reward or giveaway, a deadline, or an amount of money offered. **Any language counts.** | `yes` · `no` · `unclear` |
| `cta_language` | Only if `call_to_action` is `yes`: the language of that call to action. | `en` · `zh` · `ru` · `ko` · `ja` · `es` · `tr` · `vi` · `other:<language>`. Leave blank otherwise. |
| `purpose` | In your judgement, is the **main purpose** of these texts to get the reader to do something (visit, contact, claim, buy), rather than to name a token? | `solicit` · `name` · `unclear` |
| `notes` | Optional. Anything that made the row hard to label. | free text |

The three questions are separate, so answer each one on its own.
- A bare domain can be `destination = yes` and `purpose = name`. An example is a project whose
  brand is its web address.
- "Claim your reward" with nowhere to go is `destination = no`, `call_to_action = yes`.

## Examples

These are invented, not from the sheet.

| name | symbol | destination | call_to_action | cta_language | purpose |
|---|---|---|---|---|---|
| `Visit rewards-xyz.com to claim` | `$500` | yes | yes | en | solicit |
| `Wrapped Ether` | `WETH` | no | no | | name |
| `Curve.fi USDC/USDT` | `2CRV` | yes | no | | name |
| `领取空投 t.me/xyzdrop` | `GIFT` | yes | yes | zh | solicit |
| `Claim your bonus now` | `BONUS` | no | yes | en | solicit |

## When you are done

- **On the web page:** press **Save ratings file** on the last screen and send the saved file back.
- **In the spreadsheet:** save the sheet as CSV (UTF-8) under a new name ending in
  `_<your initials>.csv` and send it back. Please don't sort or delete rows: `row_id` is how the
  sheets are matched.
