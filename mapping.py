import os
import folium
from folium.features import GeoJsonTooltip
import branca


import read_data as rd

map_folder = 'country coordinates'
zoom_start = 1
location = [27.664827, -81.516]
location_zero = [27, 0]


def update_coordinates(df_COVID_summary):
    country_codes = df_COVID_summary['code']
    putFile = "country coordinates\\"
    inFile = "coords\\"

    # Add properties from column to geo
    for i in range(len(df_COVID_summary)):
        in_file_name = rd.get_relative_path(inFile + str(country_codes[i]) + ".geo.json")
        #print(in_file_name)
        if os.path.exists(in_file_name):
            text = open(in_file_name, 'r', encoding='utf-8-sig').read()
            #print(in_file_name)
            k = text.find("\"Country")
            l = text.find("},\"geometry")
            str1 = text[0:k]
            str2 = text[l:len(text)]

            str1 += "\"" + "Country" + "\":\"" + str(df_COVID_summary['Country_Region'][i]) + "\","
            str1 += "\"" + "Mesoregion" + "\":\"" + str(df_COVID_summary['meso_region'][i]) + "\","
            str1 += "\"" + "Macroregion" + "\":\"" + str(df_COVID_summary['macro_region'][i]) + "\","
            str1 += "\"" + "Total confirmed" + "\":\"" + str(df_COVID_summary['Confirmed'][i]) + "\","
            str1 += "\"" + "Total deaths" + "\":\"" + str(df_COVID_summary["Deaths"][i]) + "\""

            file = str1 + str2
            out_file_name = rd.get_relative_path(putFile + df_COVID_summary['code'][i] + ".geo.json")
            #print(out_file_name)
            file1 = open(out_file_name, 'w', encoding='utf-8')
            file1.write(file)
            file1.close()
        else:
            print("File {filepath} not found...".format(filepath=in_file_name))


def make_map(df_for_map):
    name = "COVID19 total cases"
    return get_new_map(df_for_map, 'Confirmed', name)


def get_new_map(df, value_col, title):

    current_map = folium.Map(tiles=None, location=location_zero, zoom_start=zoom_start, min_zoom=1)
    folium.TileLayer(tiles='cartodbdark_matter', attr='none').add_to(current_map)

    print(df.head())
    ma = max(df[value_col])
    mi = min(df[value_col])

    colormap = branca.colormap.LinearColormap(
        vmin=int(mi),
        vmax=int(ma),
        colors=['SpringGreen', 'DeepSkyBlue', 'BlueViolet'],
        # index=[mi,(ma-mi)*0.3+mi,(ma-mi)*0.6+mi,(ma-mi)*0.8+mi,ma],
        caption=title,
    )

    colormap.add_to(current_map)
    name_country = df['code']

    value = df[value_col].tolist()

    def stile(k):
        return lambda x: {
            'fillColor': colormap(value[k]),
            "color": '#404040',
            "weight": 1,
            "fillOpacity": 0.7,
        }

    tooltip = []
    fields = ['Country', 'Mesoregion', 'Macroregion', 'Total confirmed', 'Total deaths']

    names = ['Country', 'Mesoregion', 'Macroregion', 'Total confirmed', 'Total deaths']

    for i in range(len(df)):
        tooltip.append(GeoJsonTooltip(
            fields=[*fields],
            aliases=[*names],
            localize=True,
            sticky=False,
            labels=True,
            style="""
            background-color: 'darkGray';
            color: 'white'
            border: 1px solid black;
            border-radius: 1px;
            box-shadow: 1px;
        """,
            max_width=800,
        ))

    for i in range(len(df)):
        file_name = rd.get_relative_path(str(name_country[i]) + ".geo.json", map_folder)
        if os.path.exists(file_name):
            current_map.add_child(folium.GeoJson(data=open(file_name, 'r', encoding='utf-8-sig').read(),
                                                 tooltip=tooltip[i], style_function=stile(i)))

    return current_map
