import pandas as pd
import numpy as np

def generate_leads(n=2000):
    np.random.seed(42)

    # --- Categorical columns ---
    lead_source = np.random.choice(
        ["LinkedIn", "Referral", "Website",
         "Cold Outreach", "Event"], n
    )

    ad_platform = np.random.choice(
        ["LinkedIn", "Instagram", "Facebook",
         "TikTok", "X", "YouTube", "None"], n
    )

    industry = np.random.choice(
        ["Technology", "Healthcare", "Finance",
         "Retail", "Education", "Real Estate",
         "Food & Beverage", "Fashion", "Logistics",
         "Construction", "Beauty & Wellness",
         "Legal", "Consulting", "Non-Profit",
         "Entertainment"], n
    )

    company_size = np.random.choice(
        ["Small", "Medium", "Large"], n,
        p=[0.6, 0.3, 0.1]
    )

    job_title = np.random.choice(
        ["CEO", "Founder", "Co-Founder", "Owner",
         "Managing Director", "Director", "VP of Marketing",
         "VP of Sales", "Marketing Manager", "Sales Manager",
         "Business Development Manager", "Operations Manager",
         "General Manager", "Executive", "Consultant"], n
    )

    funnel_stage = np.random.choice(
        ["Awareness", "Consideration", "Decision"], n,
        p=[0.5, 0.3, 0.2]
    )

    preferred_contact_time = np.random.choice(
        ["Morning", "Afternoon", "Evening"], n
    )

    # --- Realistic binary columns ---
    demo_requested = np.random.choice(
        [0, 1], n, p=[0.8, 0.2]
    )
    webinar_attended = np.random.choice(
        [0, 1], n, p=[0.75, 0.25]
    )
    budget_indicated = np.random.choice(
        [0, 1], n, p=[0.7, 0.3]
    )
    decision_maker = np.random.choice(
        [0, 1], n, p=[0.65, 0.35]
    )

    # --- Correlated behavioral columns ---
    # website_visits is base
    website_visits = np.random.randint(0, 50, n)

    # pages_viewed correlates with website_visits
    pages_viewed = np.clip(
        website_visits + np.random.randint(0, 5, n), 1, 60
    )

    # email_opens is base
    email_opens = np.random.randint(0, 20, n)

    # webinar attendees tend to open more emails
    email_opens = np.where(
        webinar_attended == 1,
        np.clip(email_opens + np.random.randint(2, 8, n), 0, 25),
        email_opens
    )

    # email_clicks correlates with email_opens
    email_clicks = np.clip(
        (email_opens * 0.5).astype(int) + np.random.randint(0, 3, n),
        0, 15
    )

    # demo requesters tend to have more contacts
    num_contacts = np.where(
        demo_requested == 1,
        np.random.randint(5, 20, n),
        np.random.randint(1, 10, n)
    )

    content_downloads = np.random.randint(0, 10, n)
    ad_clicks = np.random.randint(0, 15, n)
    days_since_last_contact = np.random.randint(1, 180, n)

    # --- Randomized created_date ---
    created_date = pd.to_datetime(
        np.random.choice(
            pd.date_range("2024-01-01", "2025-12-31"), n
        )
    )

    # lead_age_days recomputed from random dates
    lead_age_days = (pd.Timestamp.today() - created_date).days

    # --- Funnel stage weights ---
    stage_weights = np.array([
        {"Awareness": 0, "Consideration": 0.10, "Decision": 0.20}[s]
        for s in funnel_stage
    ])

    # --- Behavior-based conversion score ---
    score = (
        email_opens * 0.01 +
        email_clicks * 0.05 +
        website_visits * 0.03 +
        pages_viewed * 0.02 +
        content_downloads * 0.04 +
        ad_clicks * 0.03 +
        demo_requested * 0.30 +
        decision_maker * 0.20 +
        budget_indicated * 0.15 +
        webinar_attended * 0.10 +
        stage_weights -
        days_since_last_contact * 0.01
    )

    # Normalize to 0-1
    prob = (score - score.min()) / (score.max() - score.min())

    # Scale down to realistic conversion rate 10-25%
    prob = prob * 0.35

    # Add noise for realism
    noise = np.random.normal(0, 0.05, n)
    prob = np.clip(prob + noise, 0, 1)

    # Sample conversions based on probability
    converted = np.array([
        np.random.choice([0, 1], p=[1 - p, p]) for p in prob
    ])

    # --- Build dataframe ---
    df = pd.DataFrame({
        "lead_id": range(1, n + 1),
        "company_name": [f"Company_{i}" for i in range(1, n + 1)],
        "created_date": created_date,
        "lead_age_days": lead_age_days,
        "email_opens": email_opens,
        "email_clicks": email_clicks,
        "website_visits": website_visits,
        "pages_viewed": pages_viewed,
        "demo_requested": demo_requested,
        "content_downloads": content_downloads,
        "webinar_attended": webinar_attended,
        "industry": industry,
        "company_size": company_size,
        "job_title": job_title,
        "funnel_stage": funnel_stage,
        "budget_indicated": budget_indicated,
        "decision_maker": decision_maker,
        "days_since_last_contact": days_since_last_contact,
        "num_contacts": num_contacts,
        "lead_source": lead_source,
        "ad_clicks": ad_clicks,
        "ad_platform": ad_platform,
        "preferred_contact_time": preferred_contact_time,
        "converted": converted
    })

    # --- Inject missing values for realism ---
    for col, rate in [
        ("budget_indicated", 0.05),
        ("days_since_last_contact", 0.03),
        ("ad_platform", 0.04),
        ("job_title", 0.03)
    ]:
        mask = np.random.rand(n) < rate
        df.loc[mask, col] = np.nan

    return df