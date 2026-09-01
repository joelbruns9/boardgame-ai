from games.kingdomino.advisor_capacity_bench import summarize


def test_summary_projects_budgets_and_applies_relative_floor():
    rows = []
    for position in ("a", "b"):
        rows.append(
            {
                "arm": "80x6",
                "position_id": position,
                "repeat": 0,
                "simulations": 100,
                "elapsed_seconds": 1.0,
            }
        )
        rows.append(
            {
                "arm": "128x8+gp",
                "position_id": position,
                "repeat": 0,
                "simulations": 100,
                "elapsed_seconds": 2.5,
            }
        )
    result = summarize(rows, [15.0, 60.0], 3.0)

    assert result["80x6"]["simulations_per_second"] == 100.0
    assert result["128x8+gp"]["overall_slowdown_vs_80x6"] == 2.5
    assert result["128x8+gp"]["projected_simulations"]["15"] == 600.0
    assert result["128x8+gp"]["deployment_floor_pass"] is True
