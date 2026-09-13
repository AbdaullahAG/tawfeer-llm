# Quran reference data attribution

`quran_verses.json` contains verbatim Arabic verse strings from
[semarketir/quranjson](https://github.com/semarketir/quranjson), retrieved
from its `source/surah` directory. That project describes the corpus as 114
surahs and distributes it under the MIT License. The reference file is used
only to verify passages; ar-tokenwise never substitutes or edits a user’s
Quranic text.

For applications that require the Tanzil text specifically, provide its
verbatim verses to `QuranGuard(verses=...)`. Tanzil's terms require source
attribution and prohibit changing its text.
