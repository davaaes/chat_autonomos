from legal_rag_boe.segmenter import parse_consolidated_xml


def test_parse_consolidated_xml_extracts_current_article_without_historical_notes() -> None:
        xml = """<?xml version='1.0' encoding='utf-8'?>
<response>
    <data>
        <texto>
            <bloque id='a1' tipo='precepto' titulo='Artículo 1'>
                <version id_norma='BOE-A-2007-13409' fecha_publicacion='20070712' fecha_vigencia='20071012'>
                    <p class='articulo'>Artículo 1. Supuestos incluidos.</p>
                    <p class='parrafo'>1. Texto vigente.</p>
                    <blockquote class='soloTexto'>
                        <p class='parrafo'>Téngase en cuenta que texto histórico.</p>
                        <p class='cita_con_pleca'>Redacción anterior:</p>
                    </blockquote>
                    <p class='parrafo'>2. Segundo párrafo vigente.</p>
                </version>
            </bloque>
        </texto>
    </data>
</response>"""
        articles = parse_consolidated_xml(xml)
        assert len(articles) == 1
        assert articles[0][0] == "1"
        assert articles[0][1] == "Artículo 1. Supuestos incluidos"
        assert "Téngase en cuenta" not in articles[0][2]
        assert "Redacción anterior" not in articles[0][2]


def test_parse_alcance_filter_ranges() -> None:
    from legal_rag_boe.segmenter import is_article_in_scope, parse_alcance_filter

    assert parse_alcance_filter("Norma completa") is None

    allowed_reta = parse_alcance_filter("Artículos 305 a 333")
    assert allowed_reta is not None
    assert len(allowed_reta) == 29
    assert is_article_in_scope("305", allowed_reta) is True
    assert is_article_in_scope("333", allowed_reta) is True
    assert is_article_in_scope("1", allowed_reta) is False
    assert is_article_in_scope("318bis", allowed_reta) is True

    allowed_irpf = parse_alcance_filter("Artículos 27 a 32 y 95 a 101")
    assert allowed_irpf is not None
    assert is_article_in_scope("27", allowed_irpf) is True
    assert is_article_in_scope("32", allowed_irpf) is True
    assert is_article_in_scope("95", allowed_irpf) is True
    assert is_article_in_scope("101", allowed_irpf) is True
    assert is_article_in_scope("50", allowed_irpf) is False
