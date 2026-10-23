def calculate_priority(
    severity,
    population_affected,
    injuried_people,
    displaced_people,
    infrastructure_damage
):

    population_score = min(
        population_affected / 50000,
        1
    )

    injuried_score = min(
        injuried_people / 3000,
        1
    )

    displaced_score = min(
        displaced_people / 30000,
        1
    )

    damage_score = (
        infrastructure_damage / 100
    )

    severity_score = (
        severity / 10
    )


    priority = (

        severity_score * 0.30

        + population_score * 0.20

        + injuried_score * 0.20

        + displaced_score * 0.15

        + damage_score * 0.15

    )


    return round(
        priority * 100,
        2
    )


def priority_level(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


def allocate_resource(
    required,
    available,
    priority_score
):

    if available <= 0:
        return 0

    if required <= available:
        return required

    if priority_score >= 75:

        allocation = available * 0.80

    elif priority_score >= 50:

        allocation = available * 0.60

    elif priority_score >= 25:

        allocation = available * 0.40

    else:

        allocation = available * 0.20

    return int(allocation)