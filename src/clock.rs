//! `strftime_now` support. Templates (Llama-3.1+, Command-R) call `strftime_now("%d %B %Y")`
//! to stamp the current date into system prompts. We expose a [`Clock`] trait so tests can
//! pin the date for byte-stable golden output, and ship a dependency-free default.
//!
//! The strftime implementation is intentionally minimal: it supports the conversion
//! specifiers that real chat templates actually use. Unknown specifiers are passed through
//! verbatim (matching neither libc nor Python perfectly — documented, and the corpus is the
//! check on what's actually needed).

/// Supplies "now", formatted per a strftime-style format string. Injectable for determinism.
pub trait Clock: Send + Sync {
    /// Format the current instant per the given strftime `format`.
    fn strftime(&self, format: &str) -> String;
}

/// Civil date+time broken into fields. Internal building block shared by the clocks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Civil {
    pub year: i64,
    pub month: u32, // 1..=12
    pub day: u32,   // 1..=31
    pub hour: u32,
    pub min: u32,
    pub sec: u32,
    pub weekday: u32, // 0=Sunday .. 6=Saturday
    pub yday: u32,    // 1..=366
}

const MONTHS_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const MONTHS_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const DAYS_FULL: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const DAYS_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Convert days since the Unix epoch (1970-01-01) to a civil (year, month, day) using
/// Howard Hinnant's well-known algorithm. Valid across the full practical range.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Is `year` a leap year (proleptic Gregorian)?
fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Day-of-year (1-based) for a civil date.
fn day_of_year(year: i64, month: u32, day: u32) -> u32 {
    const CUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut d = CUM[(month - 1) as usize] + day;
    if month > 2 && is_leap(year) {
        d += 1;
    }
    d
}

impl Civil {
    /// Build civil fields from seconds since the Unix epoch (UTC).
    pub(crate) fn from_unix_secs(secs: i64) -> Civil {
        let days = secs.div_euclid(86_400);
        let tod = secs.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);
        // 1970-01-01 was a Thursday (weekday index 4 with 0=Sunday).
        let weekday = ((days.rem_euclid(7) + 4) % 7) as u32;
        Civil {
            year,
            month,
            day,
            hour: (tod / 3600) as u32,
            min: ((tod % 3600) / 60) as u32,
            sec: (tod % 60) as u32,
            weekday,
            yday: day_of_year(year, month, day),
        }
    }

    /// Render this instant with a minimal strftime supporting the specifiers chat templates use.
    pub(crate) fn strftime(&self, format: &str) -> String {
        let mut out = String::with_capacity(format.len() + 8);
        let mut chars = format.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('Y') => out.push_str(&self.year.to_string()),
                Some('y') => out.push_str(&format!("{:02}", self.year.rem_euclid(100))),
                Some('m') => out.push_str(&format!("{:02}", self.month)),
                Some('d') => out.push_str(&format!("{:02}", self.day)),
                Some('e') => out.push_str(&format!("{:2}", self.day)),
                Some('B') => out.push_str(MONTHS_FULL[(self.month - 1) as usize]),
                Some('b') | Some('h') => out.push_str(MONTHS_ABBR[(self.month - 1) as usize]),
                Some('A') => out.push_str(DAYS_FULL[self.weekday as usize]),
                Some('a') => out.push_str(DAYS_ABBR[self.weekday as usize]),
                Some('j') => out.push_str(&format!("{:03}", self.yday)),
                Some('H') => out.push_str(&format!("{:02}", self.hour)),
                Some('I') => {
                    let h12 = match self.hour % 12 {
                        0 => 12,
                        h => h,
                    };
                    out.push_str(&format!("{:02}", h12));
                }
                Some('M') => out.push_str(&format!("{:02}", self.min)),
                Some('S') => out.push_str(&format!("{:02}", self.sec)),
                Some('p') => out.push_str(if self.hour < 12 { "AM" } else { "PM" }),
                Some('%') => out.push('%'),
                // Unknown specifier: emit verbatim (`%X`), so we don't silently corrupt.
                Some(other) => {
                    out.push('%');
                    out.push(other);
                }
                None => out.push('%'),
            }
        }
        out
    }
}

/// Real wall-clock (UTC). Dependency-free: reads `SystemTime` and does the calendar math here.
///
/// Note: this is UTC, not local time. transformers uses local time via Python's `datetime.now()`.
/// For reproducible/server use UTC is usually preferable; use [`LocalClock`] (the `strftime`
/// feature) to match Python's local-time behavior, or pin a [`FixedClock`] to match a specific
/// reference exactly.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn strftime(&self, format: &str) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Civil::from_unix_secs(secs).strftime(format)
    }
}

/// A clock pinned to a fixed Unix timestamp — for deterministic golden tests.
#[derive(Clone, Copy, Debug)]
pub struct FixedClock {
    unix_secs: i64,
}

impl FixedClock {
    /// Pin to a specific number of seconds since the Unix epoch (UTC).
    pub fn from_unix_secs(unix_secs: i64) -> Self {
        FixedClock { unix_secs }
    }

    /// Pin to a specific civil date at 00:00:00 UTC. Months and days are 1-based.
    /// Returns `None` for an obviously invalid date.
    pub fn from_ymd(year: i64, month: u32, day: u32) -> Option<Self> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        // days from civil (inverse of civil_from_days), Hinnant.
        let y = if month <= 2 { year - 1 } else { year };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let mp = if month > 2 { month - 3 } else { month + 9 } as i64;
        let doy = (153 * mp + 2) / 5 + day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146_097 + doe - 719_468;
        Some(FixedClock {
            unix_secs: days * 86_400,
        })
    }
}

impl Clock for FixedClock {
    fn strftime(&self, format: &str) -> String {
        Civil::from_unix_secs(self.unix_secs).strftime(format)
    }
}

/// Real wall-clock in the **local** timezone, matching Python's `datetime.now()` (which is what
/// `transformers` uses for `strftime_now`). Requires the `strftime` feature.
///
/// [`SystemClock`] is UTC and dependency-free; this reads the local timezone offset via `chrono`
/// (the only thing we borrow from it — formatting still goes through our own strftime, so the
/// supported specifiers and their output are identical to the other clocks). Construct it and pass
/// it to [`ChatTemplateBuilder::clock`](crate::ChatTemplateBuilder::clock) when you need rendered
/// dates to match what `transformers` would emit on the same machine:
///
/// ```
/// # #[cfg(feature = "strftime")] {
/// use hf_chat_template::{ChatTemplate, LocalClock};
/// let tmpl = ChatTemplate::builder("Today: {{ strftime_now('%Y') }}")
///     .clock(LocalClock)
///     .build()
///     .unwrap();
/// # }
/// ```
#[cfg(feature = "strftime")]
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalClock;

#[cfg(feature = "strftime")]
impl Clock for LocalClock {
    fn strftime(&self, format: &str) -> String {
        use chrono::{Datelike, Local, Timelike};
        let now = Local::now();
        let civil = Civil {
            year: now.year() as i64,
            month: now.month(),
            day: now.day(),
            hour: now.hour(),
            min: now.minute(),
            sec: now.second(),
            // chrono: 0 = Sunday .. 6 = Saturday, matching our Civil convention.
            weekday: now.weekday().num_days_from_sunday(),
            yday: now.ordinal(), // chrono ordinal is 1-based, as is Civil::yday.
        };
        civil.strftime(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_clock_formats_a_known_date() {
        // 2024-07-04 00:00:00 UTC. Exercises the specifiers Granite-style templates use.
        let clk = FixedClock::from_ymd(2024, 7, 4).unwrap();
        assert_eq!(clk.strftime("%B %d, %Y"), "July 04, 2024");
        assert_eq!(clk.strftime("%Y-%m-%d"), "2024-07-04");
        assert_eq!(clk.strftime("%A"), "Thursday");
    }

    // LocalClock delegates to the local timezone; verify the wiring reads local wall time and maps
    // chrono's fields onto Civil correctly (not, say, UTC or an off-by-one weekday/ordinal). We
    // compare the date-portion against a fresh chrono::Local::now() taken in the test; the only
    // race is the midnight boundary, which we tolerate by allowing either side of it.
    #[cfg(feature = "strftime")]
    #[test]
    fn local_clock_reads_local_time() {
        use chrono::{Datelike, Local};
        let got = LocalClock.strftime("%Y-%m-%d");
        let now = Local::now();
        let same = format!("{:04}-{:02}-{:02}", now.year(), now.month(), now.day());
        let prev = (now - chrono::Duration::days(1)).date_naive();
        let prev = format!("{:04}-{:02}-{:02}", prev.year(), prev.month(), prev.day());
        assert!(
            got == same || got == prev,
            "LocalClock date {got} matched neither {same} nor {prev}"
        );
    }
}
