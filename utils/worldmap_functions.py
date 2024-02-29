import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt


def create_network(year):

    # returns a Graph object corresponding to the international relations network of the given year
    # edges have a weighted attribute which can be 1 or -1

    df = pd.read_csv("data/CoW_data/allies_and_enemies_1816_2014_iso.csv")

    df = df[df["year"] == year]  # select one particular year
    df = df[
        (df["alliance"] != 0) | (df["conflict"] != 0)
    ]  # filter entries with no link

    df["weight"] = (
        df["alliance"] + df["conflict"] + df["alliance"] * df["conflict"]
    )  # merge alliances and conflicts in a single column
    df = df.drop(columns=["alliance", "conflict"])

    for country in [
        "SLU",
        "SVG",
        "SKN",
        "SCN",
        "SWA",
    ]:  # remove countires that aren't in the location dataframe
        df = df[~(df["statea"] == country)]
        df = df[~(df["stateb"] == country)]

    # create the networks

    G = nx.from_pandas_edgelist(
        df, source="statea", target="stateb", edge_attr="weight"
    )

    return G


def get_countries_coordinates(countries_list):

    # import dataframe with coordinates of countries

    countries = pd.read_csv(
        "data/CoW_data/countries_codes_and_coordinates.csv",
        index_col="Alpha-3 code",
        usecols=[
            "Alpha-3 code",
            " Country",
            "Latitude (average)",
            "Longitude (average)",
        ],
    )

    # store the latitude and longitude (stored in df "countries") of each country in G as a dictiorary
    pos_iso = {}
    pos_name = {}

    for idx, node in enumerate(countries_list):

        pos_iso[node] = [
            countries.loc[node, "Longitude (average)"],
            countries.loc[node, "Latitude (average)"],
        ]
        pos_name[countries.loc[node, " Country"]] = [
            countries.loc[node, "Longitude (average)"],
            countries.loc[node, "Latitude (average)"],
        ]

        if node == "RUS":  # I prefer locating Russia in Moscow
            pos_iso[node] = [38, 56]
            pos_name[countries.loc[node, " Country"]] = [38, 56]

        elif node == "DEU":  # I prefer to locate Germany-Prussia in Berlin
            pos_iso[node] = [13, 52]

    return pos_iso, pos_name


def plot_world_map(ax, countries_colors_dict):

    # Plot a world map using geopandas. We can choose the color of the highlight

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # country colors
    default_color = "lightgray"
    world["color"] = world["iso_a3"].map(
        lambda x: countries_colors_dict.get(x, default_color)
    )

    world.plot(ax=ax, color=world["color"], edgecolor="black", alpha=0.5)
    ax.axis("off")
